import os
from contextlib import contextmanager

import psycopg

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "churn")
DB_USER = os.getenv("DB_USER", "app")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS churn_records (
    id SERIAL PRIMARY KEY,
    credit_score INTEGER,
    geography TEXT,
    gender TEXT,
    age INTEGER,
    tenure INTEGER,
    balance NUMERIC,
    num_of_products INTEGER,
    has_cr_card BOOLEAN,
    is_active_member BOOLEAN,
    estimated_salary NUMERIC,
    exited INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
)
"""


def get_connection():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=True,
    )


@contextmanager
def get_cursor():
    with get_connection() as conn:
        with conn.cursor() as cur:
            yield cur


def init_db():
    with get_cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)


def insert_record(payload: dict) -> int:
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO churn_records (
                credit_score,
                geography,
                gender,
                age,
                tenure,
                balance,
                num_of_products,
                has_cr_card,
                is_active_member,
                estimated_salary,
                exited
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                payload["CreditScore"],
                payload["Geography"],
                payload["Gender"],
                payload["Age"],
                payload["Tenure"],
                payload["Balance"],
                payload["NumOfProducts"],
                bool(payload["HasCrCard"]),
                bool(payload["IsActiveMember"]),
                payload["EstimatedSalary"],
                payload.get("Exited", 0),
            ),
        )
        return cur.fetchone()[0]
