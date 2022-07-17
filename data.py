import re
import codecs
from functools import partial
from pathlib import Path
from itertools import chain
from dataclasses import dataclass
from text_unidecode import unidecode
from typing import Any, Optional, Tuple

import pandas as pd
from sklearn.model_selection import (
    KFold,
)
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)
from datasets import Dataset, load_from_disk


def get_folds(df, k_folds=5):

    kf = KFold(n_splits=k_folds)
    return [
        val_idx
        for _, val_idx in kf.split(df)
    ]


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def read_text_files(example, data_dir):

    id_ = example["essay_id"]

    with open(data_dir / "train" / f"{id_}.txt", "r") as fp:
        example["text"] = resolve_encodings_and_normalize(fp.read())

    return example


def find_positions(data):

    dt = data.discourse_text
    text = data.text[0]

    idx = 0

    idxs = []

    for d in dt:
        matches = list(re.finditer(re.escape(d.strip()), text))
        if len(matches) > 1:
            for m in matches:
                if m.start() >= idx:
                    break
        elif len(matches) == 0:
            idxs.append("?")  # will filter out later
        else:
            m = matches[0]
        idxs.append((m.start(), m.end()))

        idx = m.start()

    return idxs


@dataclass
class TokenClassificationDataModule:

    cfg: dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError("Please provide a config file")

        self.label2idx = {
            "Adequate": 0,
            "Effective": 1,
            "Ineffective": 2,
        }

        disc_types = [
            "Claim",
            "Concluding Statement",
            "Counterclaim",
            "Evidence",
            "Lead",
            "Position",
            "Rebuttal",
        ]

        self.data_dir = Path(self.cfg["data_dir"])

        if self.cfg["load_from_disk"]:
            self.ds = load_from_disk((f"{self.cfg['output']}.dataset"))

        else:
            train_df = pd.read_csv(self.data_dir / "train.csv")

            text_ds = Dataset.from_dict({"essay_id": train_df.essay_id.unique()})

            text_ds = text_ds.map(
                partial(read_text_files, data_dir=self.data_dir),
                num_proc=self.cfg["num_proc"],
                batched=False,
            )

            text_df = text_ds.to_pandas()

            train_df["discourse_text"] = [
                resolve_encodings_and_normalize(x) for x in train_df["discourse_text"]
            ]

            self.train_df = train_df.merge(text_df, on="essay_id", how="left")

        self.cls_tkn_map = {label: f"[CLS_{label.upper()}" for label in disc_types}

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name_or_path"])
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(self.cls_tkn_map.values())}
        )
        self.cls_id_map = {
            label: self.tokenizer.encode(tkn)[1]
            for label, tkn in self.cls_tkn_map.items()
        }

    def prepare_datasets(self):

        if self.cfg["load_from_disk"] is None:

            grouped = self.train_df.groupby(["essay_id"]).agg(list)

            positions = grouped[["discourse_text", "text"]].apply(
                find_positions, axis=1
            )
            positions.name = "idxs"
            positions = positions.reset_index()

            grouped = grouped.merge(positions, on="essay_id", how="left")
            grouped["text"] = [g[0] for g in grouped["text"]]

            self.ds = Dataset.from_pandas(grouped)

            self.ds = self.ds.map(
                self.tokenize,
                batched=False,
                num_proc=self.cfg["num_proc"],
                desc="Tokenizing",
            )

            self.ds.save_to_disk(f"{self.cfg['output']}.dataset")

            print("Saving dataset to disk:", self.cfg["output"])
        
        self.folds = get_folds(self.ds["labels"])

    def get_train_dataset(self, fold):
        idxs = list(chain(*[i for f, i in enumerate(self.fold_idxs) if f != fold]))
        return self.ds.select(idxs)

    def get_eval_dataset(self, fold):
        idxs = self.fold_idxs[fold]
        # print("Unique eval fold values:", self.raw_ds.select(idxs).unique("fold"))
        return self.ds.select(idxs)

    def tokenize(self, example):

        text = example["text"]
        chunks = []
        labels = []
        prev = 0

        zipped = zip(
            example["idxs"],
            example["discourse_type"],
            example["discourse_effectiveness"],
        )
        for idxs, disc_type, disc_effect in zipped:
            if idxs == "?":
                continue

            s, e = idxs

            if s != prev:
                chunks.append(text[prev:s])
                prev = s
            if s == prev:
                chunks.append(self.cls_tkn_map[disc_type])
                chunks.append(text[s:e])
                chunks.append(self.tokenzier.sep_token)
            prev = e

            labels.append(self.label2idx[disc_effect])

        tokenized = self.tokenizer(
            text,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )

        tokenized["labels"] = labels

        return tokenized


@dataclass
class OnlyMaskingCollator(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>

    See original source code here: https://github.com/huggingface/transformers/blob/8b332a6a160c6df82e4267aaf118d87377d78a67/src/transformers/data/data_collator.py#L607
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # IGNORE RANDOM/NO MASK
        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return inputs, labels
