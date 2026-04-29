import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    DATA_PATH,
    RESULTS_DIR,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
)

from data_utils import load_data, check_data, make_xy


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )


def make_models():
    models = {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),

        "Random Forest Regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]
        ),

        "Neural Network Regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
    }

    return models


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    check_data(df)

    X, y = make_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = make_models()

    results = []
    prediction_df = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
    })

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse, mae, r2 = evaluate(y_test, preds)

        results.append({
            "model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        })

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        prediction_df[safe_name] = preds

    results_df = pd.DataFrame(results)

    results_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    prediction_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    print("\nModel results:")
    print(results_df)

    print("\nSaved:")
    print(RESULTS_DIR / "model_results.csv")
    print(RESULTS_DIR / "test_predictions.csv")


if __name__ == "__main__":
    main()