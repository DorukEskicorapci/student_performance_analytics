import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_PATH, FIGURES_DIR, RESULTS_DIR, TARGET


def save_score_distribution():
    df = pd.read_csv(DATA_PATH)

    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET], bins=30)
    plt.title("Distribution of Final Exam Scores")
    plt.xlabel("Final Exam Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_exam_score_distribution.png", dpi=300)
    plt.close()


def save_mean_score_by_internet():
    df = pd.read_csv(DATA_PATH)
    summary = df.groupby("internet_access")[TARGET].mean()

    plt.figure(figsize=(6, 5))
    bars = plt.bar(summary.index, summary.values)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.title("Mean Final Exam Score by Internet Access")
    plt.xlabel("Internet Access")
    plt.ylabel("Mean Final Exam Score")
    plt.ylim(60, 68)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mean_score_by_internet_access.png", dpi=300)
    plt.close()


def save_model_comparison():
    results = pd.read_csv(RESULTS_DIR / "model_results.csv")

    plt.figure(figsize=(8, 5))
    plt.bar(results["model"], results["RMSE"])
    plt.title("Model Comparison by RMSE")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_rmse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(results["model"], results["MAE"])
    plt.title("Model Comparison by MAE")
    plt.ylabel("MAE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_mae.png", dpi=300)
    plt.close()


def save_actual_vs_predicted():
    preds = pd.read_csv(RESULTS_DIR / "test_predictions.csv")

    model_columns = [col for col in preds.columns if col != "actual"]

    for col in model_columns:
        title = col.replace("_", " ").title()

        plt.figure(figsize=(6, 6))
        plt.scatter(preds["actual"], preds[col], alpha=0.4)

        min_value = min(preds["actual"].min(), preds[col].min())
        max_value = max(preds["actual"].max(), preds[col].max())

        plt.plot([min_value, max_value], [min_value, max_value])
        plt.title(f"{title}: Actual vs Predicted")
        plt.xlabel("Actual Final Exam Score")
        plt.ylabel("Predicted Final Exam Score")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{col}_actual_vs_predicted.png", dpi=300)
        plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    save_score_distribution()
    save_mean_score_by_internet()
    save_model_comparison()
    save_actual_vs_predicted()

    print("\nFigures saved to:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()