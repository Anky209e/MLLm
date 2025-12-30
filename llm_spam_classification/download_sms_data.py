import urllib.request
import ssl
import zipfile
import os
from pathlib import Path


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


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_data.zip"
    extracted_path = "sms_data"

    data_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    download_and_unzip(
        url=url, zip_path=zip_path, extracted_path=extracted_path, data_path=data_path
    )
