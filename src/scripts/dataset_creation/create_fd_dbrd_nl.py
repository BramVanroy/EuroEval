# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SST5-mini sentiment dataset and upload it to the HF Hub."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the SST5-mini sentiment dataset and upload it to the HF Hub.
    
    Difference with original EuroEval:
      - Instead of taking a subset, we take the full, original train/dev/test
    """
    # Define the base download URL
    repo_id = "benjaminvdb/dbrd"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Rename the label column to strings
    label_mapping = {0: "negative", 1: "positive"}
    train_df["label"] = train_df["label"].apply(label_mapping.get)
    test_df["label"] = test_df["label"].apply(label_mapping.get)

    # Remove samples ending with "…"
    train_df = train_df[train_df.text.map(lambda x: not x.endswith("…"))]
    test_df = test_df[test_df.text.map(lambda x: not x.endswith("…"))]

    # We impose stricter maximum sizes of documents, as many of them are too long
    max_num_chars = min(MAX_NUM_CHARS_IN_DOCUMENT, 3_500)  # noqa: F841

    # Only work with samples where the document is not very large or small
    train_df = train_df.copy()
    train_df["text_len"] = train_df.text.str.len()
    train_df = train_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @max_num_chars"
    )
    test_df = test_df.copy()
    test_df["text_len"] = test_df.text.str.len()
    test_df = test_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @max_num_chars"
    )

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    dataset_id = "BramVanroy/euroeval-dbrd-nl-full"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
