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
    return [val_idx for _, val_idx in kf.split(df)]


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
            idxs.append([-1])
            continue  # will filter out later
        else:
            m = matches[0]
        idxs.append([m.start(), m.end()])

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

        self.cls_tokens_map = {label: f"[CLS_{label.upper()}]" for label in disc_types}
        self.end_tokens_map = {label: f"[END_{label.upper()}]" for label in disc_types}

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name_or_path"])
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": list(self.cls_tokens_map.values())
                + list(self.end_tokens_map.values())
            }
        )
        self.cls_id_map = {
            label: self.tokenizer.encode(tkn)[1]
            for label, tkn in self.cls_tokens_map.items()
        }
        self.end_id_map = {
            label: self.tokenizer.encode(tkn)[1]
            for label, tkn in self.end_tokens_map.items()
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

            if self.cfg["stride_over"]:
                self.ds = self.ds.map(
                    self.chunk_sequences,
                    batched=True,
                    num_proc=self.cfg["num_proc"],
                    desc="chunking",
                )

            self.ds.save_to_disk(f"{self.cfg['output']}.dataset")

            print("Saving dataset to disk:", self.cfg["output"])

        self.fold_idxs = get_folds(self.ds["labels"])

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
            if idxs == [-1]:
                continue

            s, e = idxs

            if s != prev:
                chunks.append(text[prev:s])
                prev = s

            if s == prev:
                chunks.append(self.cls_tokens_map[disc_type])
                chunks.append(text[s:e])
                chunks.append(self.end_tokens_map[disc_type])
            prev = e

            labels.append(self.label2idx[disc_effect])

        if self.cfg["stride_over"]:
            max_length = None
            truncation = False
        else:
            max_length = self.cfg["max_length"]
            truncation = True

        tokenized = self.tokenizer(
            " ".join(chunks),
            padding=False,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=True,
        )

        # at this point, labels is not the same shape as input_ids.
        # The following loop will add -100 so that the loss function
        # ignores all tokens except CLS tokens.
        # if average_span_preds is True, then all tokens in a span will
        # be labeled, including CLS and END token

        # idx for labels list
        idx = 0
        final_labels = []
        in_span = False
        cls_ids = set(self.cls_id_map.values())
        end_ids = set(self.end_id_map.values())

        # indicator will be for identifying spans later
        indicator = []
        for id_ in tokenized["input_ids"]:
            new_label = -100

            # if this id belongs to a CLS token
            if id_ in cls_ids:
                new_label = labels[idx]
                in_span = True
                indicator.append(idx)

            # if this id belongs to a END token
            elif id_ in end_ids:
                # When averaging over a span, all tokens in the span
                # will have the same label
                if self.cfg["average_span_preds"]:
                    new_label = labels[idx]
                indicator.append(idx)
                in_span = False
                idx += 1

            # every token in a span should have the same label
            elif in_span:
                # When averaging over a span, all tokens in the span
                # will have the same label
                if self.cfg["average_span_preds"]:
                    new_label = labels[idx]
                indicator.append(idx)
            else:
                # -100 will be ignored by loss function
                indicator.append(-100)

            final_labels.append(new_label)

        tokenized["labels"] = final_labels
        tokenized["indicator"] = indicator

        return tokenized

    def chunk_sequences(self, sequences):
        """

        0:max_length
        max_length-stride:2max_length-stride
        2max_length-2stride:3max_length-2stride
        """
        
        to_return = {k: [] for k in sequences.keys()}

        for ids, mask, labels, indicator in zip(
            sequences["input_ids"],
            sequences["attention_mask"],
            sequences["labels"],
            sequences["indicator"]
        ):
            chunked = {k: [] for k in sequences.keys()}

            max_len = self.cfg["max_length"]-2
            stride = self.cfg["stride"]

            start = 0
            end = start+max_len
            while True:

                prepend = {
                        "input_ids": [self.tokenizer.cls_token_id],
                        "attention_mask": [1],
                        "labels": [-100],
                        "indicator": [-100],
                    }
                append = {
                    "input_ids": [self.tokenizer.eos_token_id],
                    "attention_mask": [1],
                    "labels": [-100],
                    "indicator": [-100],
                }
                if start == 0:
                    # There is already a CLS token
                    prepend = {
                        "input_ids": [],
                        "attention_mask": [],
                        "labels": [],
                        "indicator": []
                    }
                if end >= len(ids):
                    # There is already an EOS token
                    append = {
                        "input_ids": [],
                        "attention_mask": [],
                        "labels": [],
                        "indicator": []
                    }

            
                chunked["input_ids"].append(
                    prepend["input_ids"] + ids[start:end] + append["input_ids"]
                )
                chunked["attention_mask"].append(
                    prepend["attention_mask"] + mask[start:end] + append["attention_mask"]
                )
                chunked["labels"].append(
                    prepend["labels"] + labels[start:end] + append["labels"]
                )
                chunked["indicator"].append(
                    prepend["indicator"] + indicator[start:end] + append["indicator"]
                )

                start = end - stride
                if end >= len(ids):
                    break
                end = start + max_len
                    
            for k in chunked.keys():
                to_return[k].extend(chunked[k])

        return chunked


@dataclass
class ComparisonDataModule:

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

        self.cls_tokens_map = {label: f"[CLS_{label.upper()}]" for label in disc_types}
        self.end_tokens_map = {label: f"[END_{label.upper()}]" for label in disc_types}

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name_or_path"])
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": list(self.cls_tokens_map.values())
                + list(self.end_tokens_map.values())
            }
        )
        self.cls_id_map = {
            label: self.tokenizer.encode(tkn)[1]
            for label, tkn in self.cls_tokens_map.items()
        }
        self.end_id_map = {
            label: self.tokenizer.encode(tkn)[1]
            for label, tkn in self.end_tokens_map.items()
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

        self.fold_idxs = get_folds(self.ds["labels"])

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
            if idxs == [-1]:
                continue

            s, e = idxs

            if s != prev:
                chunks.append(text[prev:s])
                prev = s

            if s == prev:
                chunks.append(self.cls_tokens_map[disc_type])
                chunks.append(text[s:e])
                chunks.append(self.end_tokens_map[disc_type])
            prev = e

            labels.append(self.label2idx[disc_effect])

        tokenized = self.tokenizer(
            " ".join(chunks),
            padding=False,
            truncation=True,
            max_length=self.cfg["max_length"],
            add_special_tokens=True,
        )

        # at this point, labels is not the same shape as input_ids.
        # The following loop will add -100 so that the loss function
        # ignores all tokens except CLS tokens.
        # if average_span_preds is True, then all tokens in a span will
        # be labeled, including CLS and END token

        # idx for labels list
        idx = 0
        final_labels = []
        in_span = False
        cls_ids = set(self.cls_id_map.values())
        end_ids = set(self.end_id_map.values())

        # indicator will be for identifying spans later
        indicator = []
        for id_ in tokenized["input_ids"]:
            new_label = -100

            # if this id belongs to a CLS token
            if id_ in cls_ids:
                new_label = labels[idx]
                in_span = True
                indicator.append(idx)

            # if this id belongs to a END token
            elif id_ in end_ids:
                # When averaging over a span, all tokens in the span
                # will have the same label
                if self.cfg["average_span_preds"]:
                    new_label = labels[idx]
                indicator.append(idx)
                in_span = False
                idx += 1

            # every token in a span should have the same label
            elif in_span:
                # When averaging over a span, all tokens in the span
                # will have the same label
                if self.cfg["average_span_preds"]:
                    new_label = labels[idx]
                indicator.append(idx)
            else:
                # -100 will be ignored by loss function
                indicator.append(-100)

            final_labels.append(new_label)

        tokenized["labels"] = final_labels
        tokenized["indicator"] = indicator

        return tokenized
