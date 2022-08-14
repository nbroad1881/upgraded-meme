import argparse
from pathlib import Path
from functools import partial

import pandas as pd
from datasets import Dataset
from bertopic import BERTopic

from data import read_text_files

model_options = [
    "sentence-transformers/",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]


def parse_args():

    parser = argparse.ArgumentParser(description="Choose model and settings for BERTopic")

    parser.add_argument(
        "--data_dir", "-d", required=True, type=str, help="Where the data is stored."
    )
    parser.add_argument(
        "--roberta",
        "-r",
        required=False,
        action="store_const",
        const="all-roberta-large-v1",
    )
    parser.add_argument(
        "--minilm", "-m", required=False, action="store_const", const="all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--mpnet", "-p", required=False, action="store_const", const="all-mpnet-base-v2"
    )
    parser.add_argument(
        "--num_proc",
        "-n",
        required=False,
        type=int,
        default=7,
        help="How many processes to use",
    )
    parser.add_argument(
        "--nr_topics",
        "-t",
        required=False,
        type=int,
        help="Number of topics to use. None lets the model decide.",
        default=None
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    
    if hasattr(args, "roberta"):
        model_name = args.roberta
    elif hasattr(args, "minilm"):
        model_name = args.minilm
    elif hasattr(args, "mpnet"):
        model_name = args.mpnet

    if model_name.startswith("sentence-transformers/"):
        model_name = model_name[len("sentence-transformers/") :]

    topic_model = BERTopic(embedding_model=model_name, nr_topics=args.nr_topics)

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "train.csv")

    text_ds = Dataset.from_dict({"essay_id": df.essay_id.unique()})

    # This also handles the encoding errors
    text_ds = text_ds.map(
        partial(read_text_files, data_dir=data_dir),
        num_proc=args.num_proc,
        batched=False,
    )

    topic, probs = topic_model.fit_transform(text_ds["text"])

    topic_df = pd.DataFrame(
        {"essay_id": text_ds["essay_id"], "topic": topic, "probs": probs}
    )

    df = df.merge(topic_df, on="essay_id", how="left")

    topic_model.save(Path(model_name) / "_topic_model")
