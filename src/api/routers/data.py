from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...database.db import init_db, insert_record

router = APIRouter()


class DataIngestRequest(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Exited: Optional[int] = Field(default=0, ge=0, le=1)


@router.on_event("startup")
def _init_db():
    init_db()


@router.post("/data", status_code=201)
def ingest(record: DataIngestRequest):
    payload = record.dict()
    try:
        record_id = insert_record(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"id": record_id, "message": "stored"}
