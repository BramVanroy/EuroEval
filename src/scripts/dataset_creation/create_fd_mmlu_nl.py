# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the MMLU-mini datasets and upload them to the HF Hub."""

from collections import Counter

import pandas as pd
from constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

LANGUAGES = [
    "nl",
]


MMLUX_SUBSET_IDS = {"pt": "PT-PT", "sl": "SL"}

# Define a helper function to process a single DataFrame
def process_split(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    df = df.copy() # Good practice to avoid SettingWithCopy warnings
    
    # Rename the columns
    df.rename(columns=dict(answer="label"), inplace=True)

    # Remove the samples with overly short or long texts
    df = df[
        (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & (df.option_a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_c.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_c.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_d.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_d.str.len() <= MAX_NUM_CHARS_IN_OPTION)
    ]

    def is_repetitive(text: str) -> bool:
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df[
        ~df.instruction.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
        & ~df.option_c.apply(is_repetitive)
        & ~df.option_d.apply(is_repetitive)
    ]

    # Extract the category as a column
    df["category"] = df["id"].str.split("/").str[0]

    # Make a `text` column with all the options in it
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING[lang]}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip() + "\n"
        "c. " + row.option_c.replace("\n", " ").strip() + "\n"
        "d. " + row.option_d.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.str.lower()

    # Only keep the `text`, `label` and `category` columns
    df = df[["text", "label", "category"]]

    # Remove duplicates and reset index
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def main() -> None:
    """Create the MMLU-mini datasets and upload them to the HF Hub.

    Difference with original EuroEval:
      - EuroEval concatenates the original train/test/dev splits (seems problematic to me),
        then does processing, then splits (with stratification) a smaller subset

    Raises:
        ValueError: If the number of repetitions is too high.
    """
    # Define the base download URL
    repo_id = "alexandrainst/m_mmlu"

    for language in LANGUAGES:
        # Download the dataset
        dataset = load_dataset(path=repo_id, name=language, token=True)
        assert isinstance(dataset, DatasetDict)

        # Convert the dataset to a dataframe
        train_df = dataset["train"].to_pandas()
        val_df = dataset["val"].to_pandas()
        test_df = dataset["test"].to_pandas()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        train_df = process_split(train_df, language)
        val_df = process_split(val_df, language)
        test_df = process_split(test_df, language)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
            }
        )
        
        dataset_id = "BramVanroy/euroeval-mmlu-nl-full"

        # Remove the dataset from Hugging Face Hub if it already exists
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
