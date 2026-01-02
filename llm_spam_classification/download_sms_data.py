import urllib.request
import ssl
import zipfile
import os
from pathlib import Path
import pandas as pd


def download_and_unzip(url, zip_path, extracted_path, data_path):
    if data_path.exists():
        print(f"{data_path} already exists. Skipping Download.")
        return

    ssl_context = ssl._create_unverified_context()
    print("Downloading File...")
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(zip_path, "wb") as f:
            f.write(response.read())

    print("Unzipping Data...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    print("Renaming file..")
    original_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_path, data_path)
    print(f"File downloaded and saved as {data_path}")


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=2929)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    return balanced_df


def random_split(df, train_frac, vaildation_frac):
    df = df.sample(frac=1, random_state=2929).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * vaildation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_data.zip"
    extracted_path = "sms_data"

    data_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    download_and_unzip(
        url=url, zip_path=zip_path, extracted_path=extracted_path, data_path=data_path
    )
    df = pd.read_csv(data_path, sep="\t", header=None, names=["Label", "Text"])
    print(df.head(7))
    balanced_df = create_balanced_dataset(df)
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    print(len(train_df))
    print(len(validation_df))
    print(len(test_df))
    train_df.to_csv("sms_data/train.csv", index=None)
    validation_df.to_csv("sms_data/validation.csv", index=None)
    test_df.to_csv("sms_data/test.csv", index=None)
