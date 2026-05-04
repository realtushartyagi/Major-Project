import pandas as pd
import os
import numpy as np

# Use absolute paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_urls.csv")

def preprocess():
    combined_data = []

    # Check for PhishTank
    pt_path = os.path.join(DATA_DIR, "phishtank.csv")
    if os.path.exists(pt_path):
        df = pd.read_csv(pt_path)
        urls = df['url'].tolist()
        combined_data.extend([(u, 1) for u in urls]) # 1 for Phish

    # Check for OpenPhish
    op_path = os.path.join(DATA_DIR, "openphish.txt")
    if os.path.exists(op_path):
        with open(op_path, 'r') as f:
            urls = f.read().splitlines()
        combined_data.extend([(u, 1) for u in urls])

    # Handle Missing Data / Synthetic for Demo
    if not combined_data:
        print("No raw data found. Creating synthetic samples for demonstration...")
        synthetic = [
            ("http://secure-login-bank.com", 1),
            ("https://google-accounts-verification.xyz", 1),
            ("http://paypal-security-update.io", 1),
            ("https://google.com", 0),
            ("https://github.com", 0),
            ("https://stackoverflow.com", 0),
            ("http://amaz0n-prime-reward.support", 1),
            ("https://microsoft.com/en-us/office", 0)
        ]
        combined_data.extend(synthetic)

    df_final = pd.DataFrame(combined_data, columns=['url', 'label'])
    df_final = df_final.drop_duplicates()
    df_final.to_csv(PROCESSED_FILE, index=False)
    print(f"Pre-processing complete. Saved {len(df_final)} samples to {PROCESSED_FILE}")

if __name__ == "__main__":
    preprocess()
