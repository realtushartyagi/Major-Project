import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append("../backend")
from model_utils import URLTokenizer
from adversarial_engine import AdversarialEngine

MODEL_PATH = "../models/url_classifier.h5"
TOKENIZER_PATH = "../models/tokenizer.json"
DATA_PATH = "../data/processed_urls.csv"

def run_benchmark():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train the model first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = URLTokenizer.load(TOKENIZER_PATH)
    engine = AdversarialEngine(model, tokenizer)
    
    df = pd.read_csv(DATA_PATH)
    urls = df['url'].tolist()
    labels = df['label'].values
    
    # 1. Baseline Performance
    X = tokenizer.transform(urls)
    loss, acc = model.evaluate(X, labels, verbose=0)
    print(f"Baseline Accuracy (Clean): {acc*100:.2f}%")
    
    # 2. Adversarial Performance (Homoglyph)
    adv_urls = [engine.homoglyph_attack(u) for u in urls]
    X_adv = tokenizer.transform(adv_urls)
    loss_adv, acc_adv = model.evaluate(X_adv, labels, verbose=0)
    print(f"Adversarial Accuracy (Homoglyph): {acc_adv*100:.2f}%")
    
    # 3. Robustness Drop
    drop = (acc - acc_adv) * 100
    print(f"Accuracy Drop: {drop:.2f}%")
    if drop < 10:
        print("RESULT: High Robustness achieved!")
    else:
        print("RESULT: Model vulnerable to character perturbations.")

if __name__ == "__main__":
    run_benchmark()
