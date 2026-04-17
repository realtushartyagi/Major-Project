# Sentinel-ML Deployment Guide

This guide provides a comprehensive, step-by-step procedure to deploy the **Sentinel-ML** project to a production environment (e.g., VPS, Heroku, Render, or Docker).

---

## 📂 1. Required Folders for Deployment

When preparing your deployment package, you only need the following directories. Including unnecessary folders like `venv` or `data` will significantly increase your build time and package size.

| Folder | Required? | Purpose |
| :--- | :---: | :--- |
| `backend/` | **YES** | Contains the FastAPI server, logic, and `requirements.txt`. |
| `frontend/` | **YES** | Contains `index.html` and assets served by the backend. |
| `models/` | **YES** | **Critical**: Contains the `.h5` and `.pkl` model weights. |
| `data/` | NO | Used for training/reference only. Not used at runtime. |
| `scripts/` | NO | Development/Benchmarking utilities. |
| `venv/` | **NEVER** | Should be recreated on the target server. |

---

## 🛠️ 2. Step-by-Step Deployment Procedure

### Step 1: Environment Preparation
Ensure your production server has **Python 3.9+** installed.

### Step 2: Upload Files
Upload the `backend/`, `frontend/`, and `models/` folders to your server. 
> [!TIP]
> Use a `.gitignore` file to exclude `venv`, `__pycache__`, and `data` if you are using Git to deploy.

### Step 3: Server Setup
Run the following commands on your production server:

```bash
# 1. Create a fresh virtual environment
python -m venv venv

# 2. Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# 3. Install production dependencies
pip install -r backend/requirements.txt
```

### Step 4: Run the Application
In a production environment, it is highly recommended to use **Gunicorn** with **Uvicorn workers** for better performance and stability.

1. Install Gunicorn (for Linux servers):
   ```bash
   pip install gunicorn
   ```

2. Start the server:
   ```bash
   cd backend
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
   ```
   *Note: `-w 4` specifies 4 worker processes. Adjust based on your server's CPU cores.*

---

## 🚀 3. Platform-Specific Quick Start

### Option A: Render / Railway / Fly.io (Recommended)
1. **Build Command**: `pip install -r backend/requirements.txt`
2. **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
3. **Root Directory**: Set to project root to ensure file paths (`../models`, `../frontend`) resolve correctly.

### Option B: Docker (Containerization)
Create a `Dockerfile` in the root directory:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirement files first
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary directories
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/

# Set working directory to backend to run the app
WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🌐 5. Advanced: Decoupled Deployment (Render + Vercel)

If you want to host the **Backend on Render** and the **Frontend on Vercel**, follow these specific steps.

### Step 1: Backend Deployment (Render)
1.  **Create a New Web Service**: Connect your GitHub repository.
2.  **Environment**: Select `Python`.
3.  **Root Directory**: Leave as `.` (project root).
4.  **Build Command**: `pip install -r backend/requirements.txt`
5.  **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6.  **Environment Variables**: 
    - Ensure you have enough memory (at least 2GB Starter instance recommended).
    - **PYTHON_VERSION**: Set to `3.12.0` (or ensure the `.python-version` file is in your root).

> [!IMPORTANT]
> **Python Version**: Machine learning libraries like TensorFlow do not yet support Python 3.14. I have added a `.python-version` file to the root to force Render to use **Python 3.12**.

### Step 2: Update Frontend API URL
Before deploying the frontend, you must tell it where your Render backend is located.
1.  Open `frontend/index.html`.
2.  Locate `const API_BASE_URL = "";`.
3.  Replace it with your Render service URL (e.g., `https://sentinel-backend.onrender.com`).

### Step 3: Frontend Deployment (Vercel)
1.  **Create a New Project**: Connect your GitHub repository.
2.  **Framework Preset**: Select `Other`.
3.  **Root Directory**: Leave as `.` (project root).
4.  **Output Directory**: Set to `frontend`.
5.  **Build Command**: Leave empty.

---

## ⚠️ 6. Critical Production Checklist

> [!IMPORTANT]
> - **CORS**: I have already enabled `CORSMiddleware` in `backend/main.py`. This allows your Vercel frontend to talk to your Render backend securely.
> - **Model Paths**: In `main.py`, models are loaded relative to the file. Ensure the `models/` folder is uploaded to Render alongside the `backend/` folder.
> - **Memory**: TensorFlow and PyTorch can be memory-heavy. If your Render service crashes with an "Out of Memory" error, upgrade to a larger instance.
