import os
import re

import requests
from bs4 import BeautifulSoup

# URL of the Top 100 list
url = "https://www.gutenberg.org/browse/scores/top-en.php"

# Create a directory for the books
folder_name = "Gutenberg_Top_100"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

print("Fetching list of top books...")
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Locate the 'Top 100 EBooks last 30 days' section
# Note: The id usually corresponds to the section, e.g., 'books-last-30-days'
header = soup.find("h2", string=re.compile("Top 100 EBooks last 30 days"))
if not header:
    print("Could not find the specific header. Downloading from the first list found.")
    ol = soup.find("ol")
else:
    ol = header.find_next("ol")

books = ol.find_all("li")

print(f"Found {len(books)} books. Starting download...")

for i, book in enumerate(books, 1):
    try:
        link = book.find("a")
        if not link:
            continue

        title = link.text.strip()
        # Extract the Book ID from the href (e.g., "/ebooks/84")
        href = link["href"]
        book_id = href.split("/")[-1]

        # Construct the download URL for the Plain Text (UTF-8) version
        # This URL pattern generally redirects to the correct file
        download_url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

        # create a valid filename
        valid_title = "".join(x for x in title if x.isalnum() or x in " -_").strip()
        file_path = os.path.join(folder_name, f"{valid_title}.txt")

        print(f"[{i}/100] Downloading: {title}...")

        # Download the file
        file_response = requests.get(download_url)

        # Save to file
        with open(file_path, "wb") as f:
            f.write(file_response.content)

    except Exception as e:
        print(f"Failed to download {title}: {e}")

print("Download complete!")
