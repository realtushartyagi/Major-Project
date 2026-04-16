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

## ⚠️ 4. Critical Production Checklist

> [!IMPORTANT]
> - **CORS**: The current frontend uses relative paths (e.g., `fetch('/analyze')`). This works perfectly if the frontend and backend are served from the same domain.
> - **Model Paths**: Ensure the `models/` folder is at the same level as `backend/` and `frontend/` so the relative paths in `main.py` resolve correctly.
> - **Memory**: TensorFlow and PyTorch can be memory-heavy. Ensure your production server has at least **2GB of RAM**.
