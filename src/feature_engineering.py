"""Feature engineering and chronological dataset splitting."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATE_COLS,
    TARGET_COLUMN,
    IDENTIFIER_COLS,
)

# Core engineered features
ENGINEERED_NUMERIC_COLS = [
    "lead_time_days",
    "log_lead_time",
    "distance_to_clinic_km",
    "prior_appointments_count",
    "prior_no_show_rate",
    "sms_reminder_opt_in",
    "is_first_appointment",
    "high_lead_time_flag",
    "far_distance_flag",
]

ENGINEERED_CATEGORICAL_COLS = [
    "department",
    "booking_channel",
    "insurance_type",
    "time_of_day_slot",
    "day_of_week",
    "age_group",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive operational non-clinical features known at prediction time."""
    data = df.copy()

    # Log transform lead time to moderate heavy right tail
    data["log_lead_time"] = np.log1p(np.maximum(data["lead_time_days"], 0))

    # Binary flags for business risk factors
    data["is_first_appointment"] = (data["prior_appointments_count"] == 0).astype(int)
    data["high_lead_time_flag"] = (data["lead_time_days"] >= 14).astype(int)
    data["far_distance_flag"] = (data["distance_to_clinic_km"] >= 15.0).astype(int)

    return data


def chronological_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform strict chronological train/val/test splitting based on appointment datetime.

    Guarantees no future appointment data is available to train earlier periods.
    """
    sorted_df = df.sort_values(by="appointment_datetime").reset_index(drop=True)
    n = len(sorted_df)

    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:n_train].copy()
    val_df = sorted_df.iloc[n_train:n_val].copy()
    test_df = sorted_df.iloc[n_val:].copy()

    return train_df, val_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for numerical scaling and one-hot encoding."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ENGINEERED_NUMERIC_COLS),
            ("cat", categorical_transformer, ENGINEERED_CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_and_save_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process raw dataset, engineer features, create chronological splits, and save CSVs."""
    raw_path = RAW_DATA_DIR / "appointments_synthetic.csv"
    if not raw_path.exists():
        from src.data_generator import save_synthetic_data
        save_synthetic_data()

    raw_df = pd.read_csv(raw_path)
    df_feat = add_engineered_features(raw_df)

    train_df, val_df, test_df = chronological_train_val_test_split(df_feat)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Splits saved to {PROCESSED_DATA_DIR}:")
    print(f"  Train: {len(train_df)} rows ({train_df['appointment_datetime'].min()} to {train_df['appointment_datetime'].max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['appointment_datetime'].min()} to {val_df['appointment_datetime'].max()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['appointment_datetime'].min()} to {test_df['appointment_datetime'].max()})")
    print(f"  Target positive rate: Train={train_df[TARGET_COLUMN].mean():.2%}, Val={val_df[TARGET_COLUMN].mean():.2%}, Test={test_df[TARGET_COLUMN].mean():.2%}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_and_save_splits()
