import pandas as pd
import numpy as np
import os
from model_utils import URLTokenizer, build_model, MAX_LEN, VOCAB_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_urls.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "url_classifier.h5")
LR_MODEL_PATH = os.path.join(BASE_DIR, "models", "lr_model.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.json")

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Data file {DATA_PATH} not found. Running pre-processing first...")
        import sys
        sys.path.append("../scripts")
        from preprocess import preprocess
        preprocess()

    df = pd.read_csv(DATA_PATH)
    urls = df['url'].astype(str).tolist()
    labels = df['label'].values

    print(f"Training on {len(urls)} samples...")

    tokenizer = URLTokenizer(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    tokenizer.fit(urls)
    tokenizer.save(TOKENIZER_PATH)

    X = tokenizer.transform(urls)
    y = labels

    # 1. Train CNN (Deep Learning)
    print("Training CNN model...")
    model = build_model(VOCAB_SIZE, max_len=MAX_LEN)
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
    model.save(MODEL_PATH)
    print(f"CNN Model saved to {MODEL_PATH}")

    # 2. Train Logistic Regression
    from sklearn.linear_model import LogisticRegression
    import pickle
    
    print("Training Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    with open(LR_MODEL_PATH, 'wb') as f:
        pickle.dump(lr, f)
    
    # 3. Train Random Forest
    from sklearn.ensemble import RandomForestClassifier
    print("Training Random Forest baseline...")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf, f)

    print("All models trained and saved.")

if __name__ == "__main__":
    train()
