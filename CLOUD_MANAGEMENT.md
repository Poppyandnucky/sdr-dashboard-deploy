# 🏢 Google Cloud Project & Multi-App Management Guide

This reference guide outlines how to create new production projects, initialize their infrastructure, and toggle your MacBook Pro's PyCharm terminal between different environments (e.g., your testing sandbox vs. production apps).

---

## 🆕 Phase 1: Creating a Brand-New Project

Perform these steps via your web browser when you want to spin up a completely separate, independent Streamlit application space.

1. Go to the [Google Cloud Console](https://google.com).
2. Click the project selection dropdown in the top-left menu bar.
3. Select **New Project** and choose a unique ID (e.g., `streamlit-production-prod`).
4. **Crucial:** Go to the **Billing** section of the console and link your payment method to this specific new project. (Projects do not automatically inherit billing profiles).

---

## 🛠️ Phase 2: One-Time Infrastructure Provisioning

Once the project exists in your web browser, open the project directory on your MacBook Pro inside PyCharm, open the terminal, and run these commands **once** to build the backend engines.

### 1. Point the Terminal to Your New Project
```bash
gcloud config set project YOUR_NEW_PROJECT_ID_HERE
gcloud config set run/region us-central1
```

### 2. Enable Google Compute & Deployment Engines
```bash
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com
```

### 3. Create the Target Repository Folder
*Your `deploy.sh` script relies on this repository existing to store your compiled container images.*
```bash
gcloud artifacts repositories create streamlit-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for Streamlit"
```

### 4. Authorize Docker Linkages
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

## 🔄 Phase 3: Project Synchronization (`deploy.sh` & `gcloud_configure.sh`)

Before hitting deploy on a new project, you must update the local script variables so they point to the correct cloud destination.

1. Open `deploy.sh` and change the project variable:
   ```bash
   PROJECT_ID="YOUR_NEW_PROJECT_ID_HERE"
   ```
2. Open `gcloud_configure.sh` and change the project variable:
   ```bash
   PROJECT_ID="YOUR_NEW_PROJECT_ID_HERE"
   ```
3. Run the deployment pipeline in your PyCharm terminal:
   ```bash
   chmod +x deploy.sh gcloud_configure.sh
   ./deploy.sh && ./gcloud_configure.sh
   ```

---

## 🎛️ Phase 4: Switching Environment Contexts

Because all projects live safely inside separate compartments under your `tigusa@jhu.edu` profile, your MacBook Pro can jump between workspaces instantly using a single command.

* **Switch to your Testing Sandbox:**
  ```bash
  gcloud config set project streamlit-tigusa
  ```
* **Switch to your Production App:**
  ```bash
  gcloud config set project streamlit-sdr
  ```

*To verify which project your terminal is currently controlling, run:*
```bash
gcloud config get-value project
```

## 🛠️ Phase 5: Local Docker

* **Setup and run (-rm removes containers after being used):**
```bash
docker build -t streamlit-debug:local .
docker run -p 8501:8501 -v "$(pwd)":/app streamlit-SDR:local
docker run --rm -p 8501:8501 -v "$(pwd)":/app streamlit-debug:local
```

* **Make sure port is not in used:**
```bash
lsof -i :8501  
kill -9 12345
```

* **Check what is running/stored, safe clean up, all clean up**
```bash
docker ps -a
docker volume ls
docker image prune
docker volume prune

docker image prune -a
docker container prune
docker system prune --volumes
```