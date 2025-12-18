import os
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PIL import Image
from io import BytesIO

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "dataset/images"
CSV_PATH = "dataset/image_metadata.csv"

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------

def is_valid_image(response):
    try:
        img = Image.open(BytesIO(response.content))
        img.verify()
        return True
    except Exception:
        return False

def extract_primary_image(landing_page_url):
    try:
        response = requests.get(landing_page_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Check for Open Graph image
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"):
            return urljoin(landing_page_url, og_image["content"])

        # Fallback: first <img> tag
        img_tag = soup.find("img")
        if img_tag and img_tag.get("src"):
            return urljoin(landing_page_url, img_tag["src"])

    except Exception:
        return None

    return None

def download_image(image_url, image_id):
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200 and is_valid_image(response):
            image_path = os.path.join(DATASET_DIR, f"{image_id}.jpg")
            with open(image_path, "wb") as f:
                f.write(response.content)
            return image_path
    except Exception:
        return None

    return None

# -----------------------------
# Main Processing
# -----------------------------

def build_image_dataset(landing_page_urls):
    records = []

    for idx, url in enumerate(landing_page_urls):
        print(f"Processing URL {idx+1}: {url}")
        image_url = extract_primary_image(url)

        if image_url:
            image_path = download_image(image_url, idx)
            if image_path:
                records.append([idx, url, image_url, image_path])

    # Save metadata CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "landing_page_url", "image_url", "image_path"])
        writer.writerows(records)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    landing_pages = [
        "https://example.com"  # Replace with actual ad landing pages
    ]
    build_image_dataset(landing_pages)
