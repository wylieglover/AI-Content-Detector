import os
import requests
from bs4 import BeautifulSoup
import time
import re
from tqdm import tqdm
import logging
import csv

# Define constants
TOP_100_URL = "https://www.gutenberg.org/browse/scores/top"
BASE_URL = "https://www.gutenberg.org"
SAVE_DIRECTORY = "gutenberg_top_100"
METADATA_FILE = "metadata.csv"

# Create the save directory if it doesn't exist
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

# Setup logging
logging.basicConfig(filename='download_errors.log',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.ERROR)

def sanitize_filename(name):
    """
    Sanitizes the book title to create a valid filename.
    Removes or replaces characters that are invalid in filenames.
    """
    # Remove any character that is not alphanumeric, space, hyphen, or underscore
    sanitized = re.sub(r'[^\w\s\-]', '', name)
    # Replace spaces with underscores
    sanitized = sanitized.strip().replace(' ', '_')
    return sanitized

def get_top_100_ebooks_links():
    """
    Fetches the Top 100 EBooks page and extracts the links to each book's main page.
    Returns a list of tuples containing the book title and its URL.
    """
    response = requests.get(TOP_100_URL)
    if response.status_code != 200:
        print(f"Failed to retrieve Top 100 page. Status code: {response.status_code}")
        logging.error(f"Failed to retrieve Top 100 page. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all <h2> tags to locate the "Top 100 EBooks" section
    h2_tags = soup.find_all('h2')
    top_100_ebooks_section = None

    for h2 in h2_tags:
        if 'Top 100 EBooks' in h2.text:
            top_100_ebooks_section = h2
            break

    if not top_100_ebooks_section:
        print("Couldn't find the Top 100 EBooks section.")
        logging.error("Couldn't find the Top 100 EBooks section.")
        return []

    # The <ol> list immediately after the <h2> contains the books
    ol_tag = top_100_ebooks_section.find_next_sibling('ol')
    if not ol_tag:
        print("Couldn't find the ordered list of Top 100 EBooks.")
        logging.error("Couldn't find the ordered list of Top 100 EBooks.")
        return []

    book_links = []

    for li in ol_tag.find_all('li'):
        a_tag = li.find('a')
        if a_tag and 'href' in a_tag.attrs:
            book_title = a_tag.text.strip()
            book_url = BASE_URL + a_tag['href']
            book_links.append((book_title, book_url))

    print(f"Found {len(book_links)} Top 100 EBooks.")
    return book_links

def get_plain_text_link(book_url):
    """
    Given a book's main page URL, finds the plain text (UTF-8) download link.
    Returns the absolute URL to the plain text file or None if not found.
    """
    response = requests.get(book_url)
    if response.status_code != 200:
        print(f"Failed to retrieve book page: {book_url}")
        logging.error(f"Failed to retrieve book page: {book_url} with status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Look for link with 'Plain Text UTF-8' or similar text
    link = None
    for a_tag in soup.find_all('a', href=True):
        link_text = a_tag.text.lower()
        href = a_tag['href']
        if 'plain text' in link_text and ('utf-8' in link_text or 'utf8' in link_text):
            if href.startswith('http'):
                link = href
            else:
                link = BASE_URL + href
            break

    # If not found, try to find any .txt link as a fallback
    if not link:
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.endswith('.txt') and 'utf-8' in href.lower():
                if href.startswith('http'):
                    link = href
                else:
                    link = BASE_URL + href
                break

    if not link:
        print(f"No Plain Text UTF-8 link found for {book_url}")
        logging.error(f"No Plain Text UTF-8 link found for {book_url}")
    return link

def extract_metadata(soup):
    """
    Extracts metadata from the book's main page soup.
    Returns a dictionary of metadata.
    """
    metadata = {}

    # Title
    title_tag = soup.find('h1', itemprop='name')
    if title_tag:
        metadata['Title'] = title_tag.text.strip()

    # Author(s)
    authors = []
    for author_tag in soup.find_all('a', itemprop='creator'):
        authors.append(author_tag.text.strip())
    metadata['Author'] = ', '.join(authors) if authors else 'Unknown'

    # Language
    language_tags = soup.find_all('a', href=re.compile(r'/languages/'))
    languages = [lang.text.strip() for lang in language_tags]
    metadata['Language'] = ', '.join(languages) if languages else 'Unknown'

    # Release Date
    release_tag = soup.find('th', string='Release Date')
    if release_tag:
        release_info = release_tag.find_next_sibling('td')
        if release_info:
            metadata['Release_Date'] = release_info.text.strip()

    # Subjects
    subject_tags = soup.find_all('a', href=re.compile(r'/ebooks/subject/'))
    subjects = [subject.text.strip() for subject in subject_tags]
    metadata['Subjects'] = ', '.join(subjects) if subjects else 'Unknown'

    # Bookshelf
    bookshelf_tags = soup.find_all('a', href=re.compile(r'/ebooks/bookshelf/'))
    bookshelves = [bs.text.strip() for bs in bookshelf_tags]
    metadata['Bookshelf'] = ', '.join(bookshelves) if bookshelves else 'Unknown'

    # Download Count (Optional)
    # This information might not be directly available on the page; Project Gutenberg may not provide it.
    # If available, you can extract it similarly.

    return metadata

def download_book(title, text_url, book_url, save_dir, metadata_writer):
    """
    Downloads the plain text of the book from text_url, extracts metadata, saves the text locally,
    and writes metadata to the CSV file.
    """
    try:
        response = requests.get(text_url, stream=True)
        if response.status_code == 200:
            sanitized_title = sanitize_filename(title)
            file_path = os.path.join(save_dir, f"{sanitized_title}.txt")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {title}")

            # Fetch metadata from the book's main page
            book_response = requests.get(book_url)
            if book_response.status_code == 200:
                soup = BeautifulSoup(book_response.content, 'html.parser')
                metadata = extract_metadata(soup)
                metadata['file_path'] = os.path.relpath(file_path, start=save_dir)
                metadata['label'] = 'human'  # Assuming human-written
                metadata_writer.writerow(metadata)
            else:
                print(f"Failed to retrieve metadata for {title}. Status code: {book_response.status_code}")
                logging.error(f"Failed to retrieve metadata for {title}. Status code: {book_response.status_code}")

        else:
            error_message = f"Failed to download {title}. Status code: {response.status_code}"
            print(error_message)
            logging.error(error_message)
    except Exception as e:
        error_message = f"Error downloading {title}: {e}"
        print(error_message)
        logging.error(error_message)

def main():
    book_links = get_top_100_ebooks_links()
    if not book_links:
        print("No books to download.")
        return

    # Define metadata fields based on extract_metadata function
    metadata_fields = ['Title', 'Author', 'Language', 'Release_Date', 'Subjects', 'Bookshelf', 'file_path', 'label']

    # Open the metadata CSV file
    with open(os.path.join(SAVE_DIRECTORY, METADATA_FILE), mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metadata_fields)
        writer.writeheader()

        for index, (title, url) in enumerate(tqdm(book_links, desc="Downloading Books"), start=1):
            print(f"\nProcessing {index}/{len(book_links)}: {title}")
            text_link = get_plain_text_link(url)
            if text_link:
                download_book(title, text_link, url, SAVE_DIRECTORY, writer)
            else:
                print(f"Skipping {title} due to missing Plain Text UTF-8 link.")
            # Be polite and avoid overwhelming the server
            time.sleep(1)  # Sleep for 1 second between requests

    print("\nDownload and metadata extraction process completed.")

if __name__ == "__main__":
    main()
