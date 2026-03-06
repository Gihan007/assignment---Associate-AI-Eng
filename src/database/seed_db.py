"""Utility to seed the Postgres ingestion table from the provided CSV."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from .db import init_db, insert_record, get_record_count


def normalize_payload(row: pd.Series) -> dict:
    return {
        "CreditScore": int(row["CreditScore"]),
        "Geography": str(row["Geography"]),
        "Gender": str(row["Gender"]),
        "Age": int(row["Age"]),
        "Tenure": int(row["Tenure"]),
        "Balance": float(row["Balance"]),
        "NumOfProducts": int(row["NumOfProducts"]),
        "HasCrCard": int(row["HasCrCard"]),
        "IsActiveMember": int(row["IsActiveMember"]),
        "EstimatedSalary": float(row["EstimatedSalary"]),
        "Exited": int(row.get("Exited", 0)),
    }


def seed(csv_path: Path, limit: Optional[int] = None, force: bool = False) -> int:
    df = pd.read_csv(csv_path)

    if limit is not None and limit > 0:
        df = df.head(limit)

    init_db()

    if not force:
        existing = get_record_count()
        if existing > 0:
            print(f"Skipping seed; table already has {existing} rows")
            return 0

    inserted = 0
    for _, row in df.iterrows():
        payload = normalize_payload(row)
        insert_record(payload)
        inserted += 1

    return inserted


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Seed churn_records from the dataset CSV")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/Churn_Modelling.csv"),
        help="Path to the CSV file that contains the churn dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Insert only the first LIMIT rows from the CSV",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinsert rows even if churn_records already has data",
    )
    return parser


def main() -> None:
    load_dotenv()
    parser = parse_args()
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    count = seed(csv_path, limit=args.limit, force=args.force)
    print(f"Inserted {count} rows into churn_records from {csv_path}")


if __name__ == "__main__":
    main()
