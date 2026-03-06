import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(csv_path)


def train_test_split_df(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """Split dataframe into train/test by target column."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create preprocessing pipeline with imputation, scaling, and one-hot encoding."""
    numeric_pipeline = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    categorical_pipeline = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_pipeline), numeric_features),
            ("cat", Pipeline(categorical_pipeline), categorical_features),
        ]
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple, interpretable derived features for churn signal."""
    engineered = df.copy()

    # Ratio of balance to salary; high ratio may correlate with churn risk.
    engineered["balance_salary_ratio"] = engineered["Balance"] / (engineered["EstimatedSalary"] + 1e-3)

    # Discretize age and tenure into coarse buckets to let linear models capture non-linearity.
    engineered["age_bucket"] = pd.cut(engineered["Age"], bins=[0, 30, 40, 50, 60, np.inf], labels=["<=30", "31-40", "41-50", "51-60", "60+"])
    engineered["tenure_bucket"] = pd.cut(engineered["Tenure"], bins=[-1, 2, 5, 10], labels=["0-2", "3-5", "6-10"])

    # Flag high product count as a loyalty proxy.
    engineered["multi_product"] = (engineered["NumOfProducts"] >= 2).astype(int)

    return engineered
