import os

from dotenv import load_dotenv
from fastapi import FastAPI

from .routers.data import router as data_router
from .routers.predict import router as predict_router

load_dotenv()

app = FastAPI(title="Churn Prediction API", version="0.1.0")
app.include_router(predict_router)
app.include_router(data_router)
