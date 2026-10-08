# NeuroMeet Deployment & Production Guide

This guide describes how to run and deploy NeuroMeet across local workstations, Docker containers, and cloud PaaS environments (Render, Hugging Face Spaces, Google Cloud Run).

---

## 1. Local Production Deployment

```bash
# 1. Clone repository
git clone https://github.com/your-org/NeuroMeet.git
cd NeuroMeet

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start production Uvicorn server with multiple workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## 2. Docker Containerization

NeuroMeet includes an optimized multi-stage `Dockerfile`:

```bash
# Build the Docker image
docker build -t neuromeet:latest .

# Run the container
docker run -p 8000:8000 --name neuromeet-app neuromeet:latest
```

Visit `http://localhost:8000` to interact with the Web Intelligence Studio.

---

## 3. 1-Click Cloud Deployment (Render / Cloud Run)

### Render.com
1. Create a new **Web Service** pointing to your repository.
2. Select **Docker** as the runtime environment.
3. Set the port to `8000`.
4. Deploy! Your API and Web Studio will be live at `https://your-service.onrender.com`.

### Google Cloud Run
```bash
# Build and submit container to Google Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/neuromeet:latest

# Deploy to Cloud Run with 2GB RAM
gcloud run deploy neuromeet \
  --image gcr.io/PROJECT_ID/neuromeet:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --allow-unauthenticated
```
