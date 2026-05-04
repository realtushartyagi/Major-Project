import requests
import os
import pandas as pd
import gzip
import shutil

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_phishtank():
    print("Fetching data from PhishTank...")
    # PhishTank provides a daily CSV export
    url = "http://data.phishtank.com/data/online-valid.csv"
    try:
        response = requests.get(url, stream=True)
        path = os.path.join(DATA_DIR, "phishtank.csv")
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"PhishTank data saved to {path}")
    except Exception as e:
        print(f"Error fetching PhishTank: {e}")

def fetch_openphish():
    print("Fetching data from OpenPhish...")
    url = "https://openphish.com/feed.txt"
    try:
        response = requests.get(url)
        path = os.path.join(DATA_DIR, "openphish.txt")
        with open(path, 'w') as f:
            f.write(response.text)
        print(f"OpenPhish data saved to {path}")
    except Exception as e:
        print(f"Error fetching OpenPhish: {e}")

def fetch_unizet_benign():
    print("Fetching benign URLs (Tranco Top 1M)...")
    url = "https://tranco-list.eu/top-1m.csv.zip"
    # Note: Downloading top domains as a proxy for benign URLs
    try:
        response = requests.get(url)
        zip_path = os.path.join(DATA_DIR, "tranco.zip")
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Benign URLs saved to {zip_path}")
        # In a real scenario, we'd unzip and sample
    except Exception as e:
        print(f"Error fetching benign data: {e}")

if __name__ == "__main__":
    fetch_phishtank()
    fetch_openphish()
    fetch_unizet_benign()
    print("\nData collection complete.")
    print("Note: For Kaggle data, please manually download the 'Malicious URLs Dataset' and place it in the data/ folder.")
