# Churn Prediction Mini-Project

Associate AI Engineer exam solution (Option 2: Customer Churn Prediction). This repo scaffolds an end-to-end ML workflow: data prep, model training/comparison, evaluation reporting, and an API for predictions.

## Structure
- data/ — sample or synthetic dataset (CSV) and optional generation script
- models/ — saved model artifacts (preprocessor + model)
- src/api/ — FastAPI app and routers
  - serve.py — main app
  - routers/ — predict and data endpoints
- src/database/ — Postgres connection, seeding utilities
- src/ml/ — training pipeline, data preprocessing
- requirements.txt — dependencies
- .env.example — example environment variables

## Dataset
- Using Kaggle "Bank Customer Churn Prediction" (aka Churn_Modelling.csv): https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling
- Drop identifier columns RowNumber, CustomerId, Surname. Target: Exited (1=churn, 0=stay).
- Feature set expected by API/model: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
- Place CSV at data/Churn_Modelling.csv or set TRAIN_CSV in .env.

## Planned workflow
1) Prepare data
- Load dataset, drop ID columns, optional env overrides via TRAIN_CSV / EXCLUDE_COLS.
- Preprocess with imputers, scaling, one-hot encoding for categoricals.
- Feature engineering (in pipeline): balance_salary_ratio, age_bucket, tenure_bucket, multi_product flag.

2) Train & evaluate
- Models compared: Logistic Regression (baseline), RandomForest, LightGBM, CatBoost; best F1 wins and is saved.
- Imbalance handling: SMOTE oversampling applied to training set to balance classes.
- Outputs: metrics.json (accuracy/precision/recall/F1/report), confusion_matrix.png, saved best model artifact (joblib) including preprocessing/FE.

3) Serve
- FastAPI POST /predict; request payload uses raw feature fields above; pipeline handles encoding + engineered features.
- FastAPI POST /data; inserts the same feature payload (plus optional `Exited`) into Postgres for later retraining (table `churn_records`).

4) Ops notes (brief)
- Scaling to 100k+: batch training with scikit-learn RF on m5.large or similar; consider LightGBM/XGBoost for speed, or incremental models (SGD/online LR) if streaming.
- Retraining: scheduled job (weekly/monthly) reading latest data, writing new artifact; keep versioned artifacts in S3 with checksum; use canary when swapping API models.
- Monitoring: log inputs/outputs, track drift on categorical distributions, monitor precision/recall on labeled feedback; watch serving latency.
- Cost: artifacts in S3 (Free Tier friendly), API on small container (Fargate/EC2 t4g.small) or Azure Container Apps; autoscale on CPU; batch scoring via Lambda for low-traffic.

## Setup Guide

### Prerequisites
- **Python 3.9+**: For local development (optional, since Docker handles most of it).
- **Docker & Docker Compose**: For running the application stack.
- **Git**: To clone the repository (if applicable).
- **pgAdmin or similar**: Optional, for database GUI access.
- **Postman or curl**: For testing the API.

### 1. Obtain the Project Files
- If this is a shared repository, clone it:
  ```
  git clone <repository-url>
  cd assignment
  ```
- Alternatively, ensure you have the project folder with all files (e.g., `src/`, `data/`, `docker-compose.yml`, etc.).

### 2. Environment Setup
- Copy the example environment file:
  ```
  cp .env.example .env
  ```
- Edit `.env` if needed (defaults are usually fine):
  ```
  MODEL_PATH=models/churn_model.joblib
  API_HOST=0.0.0.0
  API_PORT=8000
  TRAIN_CSV=data/Churn_Modelling.csv
  TRAIN_TARGET=Exited
  EXCLUDE_COLS=RowNumber,CustomerId,Surname
  METRICS_PATH=models/metrics.json
  DB_HOST=postgres
  DB_PORT=5432
  DB_NAME=churn
  DB_USER=app
  DB_PASSWORD=changeme
  SEED_LIMIT=100
  ```

### 3. Install Dependencies (Local Development - Optional)
If running outside Docker:
- Create a virtual environment:
  ```
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- Install packages:
  ```
  pip install -r requirements.txt
  ```

### Credentials
- **pgAdmin**: `http://localhost:5050`  
  - Email: `admin@company.com`  
  - Password: `changeme`
- **Postgres Database**: `localhost:5432`  
  - User: `app`  
  - Password: `changeme`  
  - Database: `churn`

- If services are already built, just run:
  ```
  docker compose up
  ```

### 5. Access and Test the Application
- **API Documentation**: Open `http://localhost:8008/docs` in your browser for Swagger UI.
- **Predict Endpoint**:
  - URL: `POST http://localhost:8008/predict`
  - Body (JSON):
    ```json
    {
      "CreditScore": 720,
      "Geography": "France",
      "Gender": "Male",
      "Age": 38,
      "Tenure": 5,
      "Balance": 120000.25,
      "NumOfProducts": 2,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 95000.5
    }
    ```
  - Response: `{"prediction": 0, "probability": 0.12}` (0 = no churn, 1 = churn).
- **Data Ingestion Endpoint**:
  - URL: `POST http://localhost:8008/data`
  - Body: Same as predict, plus optional `"Exited": 0`.
  - Inserts into the database.
- **Database Access**:
  - Via pgAdmin: Connect to `postgres` (host), port `5432`, DB `churn`, user `app`, password `changeme`.
  - Via CLI: `docker compose exec postgres psql -U app -d churn`
  - Check table: `\dt` or `SELECT COUNT(*) FROM churn_records;`

### 6. Retraining the Model (If Needed)
- Manually retrain:
  ```
  docker compose run --rm trainer
  ```
- This runs `python -m src.ml.train` and seeds the DB.

### 7. Testing and Validation
- Run prediction tests or use curl:
  ```
  curl -X POST http://localhost:8008/predict -H "Content-Type: application/json" -d '{"CreditScore":480,"Geography":"Germany","Gender":"Female","Age":39,"Tenure":2,"Balance":150000,"NumOfProducts":1,"HasCrCard":0,"IsActiveMember":0,"EstimatedSalary":40000}'
  ```
- Check logs: `docker compose logs api`
- Stop everything: `docker compose down`

### 8. Troubleshooting
- **Port conflicts**: Change ports in `docker-compose.yml` if 8008/5050/5432 are in use.
- **Model not loading**: Ensure `models/churn_model.joblib` exists; retrain if missing.
- **Database empty**: Run `docker compose exec api python -m src.database.seed_db --limit 1000`
- **Permission issues**: On Windows, ensure Docker has access to the project folder.

### 9. Deployment Notes
- For production, consider cloud hosting (e.g., AWS ECS, Azure Container Apps) with environment-specific `.env` files.
- Monitor with logging and alerts on prediction drift.

## Seeding Postgres
- Run `python -m src.database.seed_db` (or `docker compose exec api python -m src.database.seed_db`) to insert the rows from `data/Churn_Modelling.csv` into the `churn_records` table so pgAdmin/psql show actual activity.
- Pass `--limit 20` (or another number) to load only a few rows, add `--force` to reseed even if rows already exist, or override the CSV path via `TRAIN_CSV` in `.env` or `--csv-path`.
- When you run `docker compose up --build` the trainer service now seeds the database from the same CSV automatically (honoring `SEED_LIMIT` -- defaults to 100 -- and skipping if the table already contains rows), so a new setup should have populated data with no extra steps.

## Docker Details
- Build & run API: `docker compose up --build api`
- Run training (creates/refreshes `models/churn_model.joblib` & artifacts): `docker compose run --rm trainer`
- The Compose stack mounts `./data` and `./models`, so artifacts stay on the host and the API can serve predictions after training.
- Compose also launches a Postgres database on `postgres:5432`. The Postgres volume keeps churn records persisted between runs.
- Optionally run `docker compose up --build` to bring all services up together; `trainer` exits after training, `api` stays running, `postgres` keeps the ingestion table ready.
