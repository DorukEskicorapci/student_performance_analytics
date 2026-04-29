import pandas as pd

from config import (
    DATA_PATH,
    TARGET,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def check_data(df):
    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


def validate_columns(df):
    required = set(
        [TARGET] + LEAKAGE_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def make_xy(df):
    validate_columns(df)

    X = df.drop(columns=LEAKAGE_COLUMNS)
    y = df[TARGET]

    return X, y