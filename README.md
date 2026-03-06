# Churn Prediction Mini-Project

Associate AI Engineer exam solution (Option 2: Customer Churn Prediction). This repo scaffolds an end-to-end ML workflow: data prep, model training/comparison, evaluation reporting, and an API for predictions.

## Structure
- data/ — sample or synthetic dataset (CSV) and optional generation script
- models/ — saved model artifacts (preprocessor + model)
- src/data_prep.py — data loading/cleaning/feature engineering helpers
- src/train.py — training and evaluation entrypoint
- src/serve.py — FastAPI app including routers
- src/routers/predict.py — POST /predict endpoint
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
- Outputs: metrics.json (accuracy/precision/recall/F1/report), confusion_matrix.png, saved best model artifact (joblib) including preprocessing/FE.

3) Serve
- FastAPI POST /predict; request payload uses raw feature fields above; pipeline handles encoding + engineered features.
- FastAPI POST /data; inserts the same feature payload (plus optional `Exited`) into Postgres for later retraining (table `churn_records`).

4) Ops notes (brief)
- Scaling to 100k+: batch training with scikit-learn RF on m5.large or similar; consider LightGBM/XGBoost for speed, or incremental models (SGD/online LR) if streaming.
- Retraining: scheduled job (weekly/monthly) reading latest data, writing new artifact; keep versioned artifacts in S3 with checksum; use canary when swapping API models.
- Monitoring: log inputs/outputs, track drift on categorical distributions, monitor precision/recall on labeled feedback; watch serving latency.
- Cost: artifacts in S3 (Free Tier friendly), API on small container (Fargate/EC2 t4g.small) or Azure Container Apps; autoscale on CPU; batch scoring via Lambda for low-traffic.

## Quickstart
- Create `.env` from `.env.example` (set TRAIN_CSV if moved).
- Install deps: `pip install -r requirements.txt`
- Train: `python -m src.train` (writes model, metrics.json, confusion_matrix.png)
- Serve: `uvicorn src.serve:app --reload --port 8008`
- Test: `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"CreditScore":650,"Geography":"France","Gender":"Female","Age":35,"Tenure":5,"Balance":60000,"NumOfProducts":2,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":80000}'`

Fill in `.env` from `.env.example` before running the API.

## Docker
- Build & run API: `docker compose up --build api`
- Run training (creates/refreshes `models/churn_model.joblib` & artifacts): `docker compose run --rm trainer`
- The Compose stack mounts `./data` and `./models`, so artifacts stay on the host and the API (http://127.0.0.1:8008) can serve predictions after training.
- Compose also launches a Postgres database on `postgres:5432`. The Postgres volume keeps churn records persisted between runs.
- Optionally run `docker compose up --build` to bring all services up together; `trainer` exits after training, `api` stays running, `postgres` keeps the ingestion table ready.
