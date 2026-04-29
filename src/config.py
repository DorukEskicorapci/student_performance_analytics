from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "student_performance_data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET = "final_exam_score"

LEAKAGE_COLUMNS = [
    "student_id",
    "final_exam_score",
    "overall_score",
    "grade",
]

NUMERIC_FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignment_score",
    "midterm_score",
    "participation_score",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "internet_access",
    "extra_classes",
    "parent_education",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42