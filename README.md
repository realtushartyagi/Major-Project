# Sentinel-ML — Robust Machine Learning Defense Framework

Sentinel-ML is a sophisticated defense framework designed to protect Machine Learning-based phishing detectors from **Adversarial Attacks**. This project focuses on character-level Deep Learning and multi-layered defensive engineering.

## 🚀 Features
- **Character-level Deep Learning**: CNN-based classification for URLs.
- **Adversarial Engine**: Generates Homoglyph, Typosquatting, and FGSM attacks.
- **Multi-layered Defense**:
    - Rule-based Sanitizer.
    - Adversarial Training logic.
    - Robustness scoring and real-time validation.
- **Premium Dashboard**: A sleek "Security Terminal" UI with live analysis and threat simulation.

## 📂 Project Structure
- `backend/`: FastAPI server, model utilities, and the adversarial engine.
- `frontend/`: Single-page premium dashboard.
- `data/`: Raw and processed URL datasets.
- `models/`: Trained model binaries and tokenizers.
- `scripts/`: Data collection, pre-processing, and benchmarking scripts.

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.9+

### 2. Setup & Installation
```bash
# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Training the Model
```bash
cd backend
python train.py
```
*Note: This will automatically generate synthetic data if raw data is missing, allowing you to run the demo immediately.*

### 4. Running the Project
1. **Start the Backend API**:
   ```bash
   cd backend
   python main.py
   ```
2. **Open the Frontend**:
   Open `frontend/index.html` in any modern web browser.

### 5. Benchmarking
```bash
cd scripts
python benchmark.py
```

## 📊 Evaluation Metrics
The project tracks:
- **Accuracy (Clean)**: Model performance on standard data.
- **Accuracy (Adversarial)**: Model performance under attack.
- **Robustness Score**: Ratio of resilience (Goal > 0.90).

## 🎓 Academic Best Practices Included
- Homoglyph detection (Security).
- Adversarial Robustness Toolbox (ART) integration.
- Layered Defense architecture.
- Modular, well-documented code.
