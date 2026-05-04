# Sentinel-ML Deployment Guide

This guide details the step-by-step process to decouple the Sentinel-ML application and deploy it to a production environment. We will host the **Frontend on Vercel** and the **Backend on Render**.

---

## 1. Backend Deployment (Render)

Render is an excellent platform for hosting Python FastAPI applications, especially those requiring complex machine learning dependencies.

### Prerequisites
1. Ensure your code is pushed to a GitHub repository.
2. Verify that your machine learning models (`models/url_classifier.h5`, etc.) are included in the repository. The backend uses absolute path resolution to locate these models dynamically.

### Steps
1. Log in to [Render](https://dashboard.render.com/) and click **New +** > **Web Service**.
2. Connect your GitHub account and select your repository.
3. Use the following configuration:
   - **Name:** `sentinel-ml-api` (or your preferred name)
   - **Root Directory:** `.` (Leave blank or set to root so it has access to the `models/` directory)
   - **Environment:** `Python 3`
   - **Build Command:** `bash render-build.sh`
   - **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables:**
   - Click on "Advanced".
   - Add a new environment variable:
     - **Key:** `PYTHON_VERSION`
     - **Value:** `3.12.0`
     *(This step is critical. Sentinel-ML uses TensorFlow and ART, which require precise Python version matching to function properly in production).*
5. Click **Create Web Service**. 
6. Wait for the deployment to finish and copy your Render URL (e.g., `https://sentinel-ml-api.onrender.com`).

---

## 2. Frontend Deployment (Vercel)

Vercel is optimized for static sites and frontend frameworks. Since your frontend is a static single-page app, deploying it here ensures blazing-fast delivery.

### Connecting Frontend to Backend
Currently, your `index.html` makes relative API calls (`fetch('/analyze')`). When hosted on Vercel, we need to route these requests to your new Render backend. The cleanest way to do this without changing your JavaScript code or dealing with strict CORS policies is to use a `vercel.json` rewrite.

1. **Create a `vercel.json` file** inside your `frontend/` directory with the following content:
   ```json
   {
     "rewrites": [
       {
         "source": "/:path*",
         "destination": "https://<YOUR_RENDER_URL>/:path*"
       }
     ]
   }
   ```
   *(Replace `<YOUR_RENDER_URL>` with the URL you copied from Render).*

### Steps
1. Log in to [Vercel](https://vercel.com/) and click **Add New** > **Project**.
2. Import your GitHub repository.
3. Use the following configuration:
   - **Project Name:** `sentinel-ml-ui`
   - **Framework Preset:** `Other`
   - **Root Directory:** `frontend`
4. Leave Build and Output Settings as default (since it's purely static HTML/CSS/JS).
5. Click **Deploy**.

---

## 3. Post-Deployment Verification

1. Navigate to your new Vercel application URL.
2. The UI should load successfully.
3. Enter a target URL and click **ANALYZE TARGET**.
   - If it works, the Vercel rewrite has successfully proxied your request to the Render API!
4. Test the **GENERATE RESEARCH REPORT** button to ensure the PDF generation pipeline is functioning in the cloud environment.

### Troubleshooting
- **Build Fails on Render:** Ensure your `requirements.txt` has no conflicting versions. The `PYTHON_VERSION=3.12.0` variable usually resolves tensor/numpy compilation errors.
- **Model Path Errors:** The backend `main.py` is configured with `os.path.dirname` to resolve absolute paths. Ensure the `models/` folder was successfully pushed to GitHub alongside the `backend/` folder.
- **API Times Out:** The free tier of Render spins down after 15 minutes of inactivity. The first scan after a period of inactivity may take ~30-50 seconds as the machine learning engine cold-starts.
