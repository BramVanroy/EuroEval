# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the WikiLingua-NL-mini summarisation dataset."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the WikiLingua-NL-mini summarisation dataset and upload to HF Hub.
    
    Difference with original EuroEval:
      - Instead of taking a subset, we take the full, original train/dev/test

    """
    dataset_id = "GEM/wiki_lingua"

    dataset = load_dataset(dataset_id, "nl", token=True)
    assert isinstance(dataset, DatasetDict)

    dataset = dataset.rename_columns(
        column_mapping=dict(source="text", target="target_text")
    )

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Only work with samples where the text is not very large or small
    train_lengths = train_df.text.str.len()
    val_lengths = val_df.text.str.len()
    test_lengths = test_df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    train_df = train_df[train_lengths.between(lower_bound, upper_bound)]
    val_df = val_df[val_lengths.between(lower_bound, upper_bound)]
    test_df = test_df[test_lengths.between(lower_bound, upper_bound)]

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "BramVanroy/euroeval-wiki-lingua-nl-full"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
