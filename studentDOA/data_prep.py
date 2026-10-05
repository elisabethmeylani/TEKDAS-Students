"""Validate studentDOA.csv and create model-ready train/test files."""
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "studentDOA.csv"
TARGET = "Target"

CATEGORICAL_COLUMNS = [
    "Marital status", "Application mode", "Course", "Daytime/evening attendance",
    "Previous qualification", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International",
]
NUMERIC_COLUMNS = [
    "Application order", "Age at enrollment",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
SEMESTER_1_COLUMNS = [column for column in NUMERIC_COLUMNS if "2nd sem" not in column]
SEMESTER_2_COLUMNS = NUMERIC_COLUMNS
FEATURE_COLUMNS_SEMESTER_1 = CATEGORICAL_COLUMNS + SEMESTER_1_COLUMNS
FEATURE_COLUMNS_SEMESTER_2 = CATEGORICAL_COLUMNS + SEMESTER_2_COLUMNS
FEATURES_BY_CUTOFF = {
    "Semester 1": FEATURE_COLUMNS_SEMESTER_1,
    "Semester 2": FEATURE_COLUMNS_SEMESTER_2,
}


def load_and_validate() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    missing = sorted(set(FEATURE_COLUMNS + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    if df.empty:
        raise ValueError("Dataset kosong")
    if df[TARGET].isna().any():
        df = df.dropna(subset=[TARGET]).copy()
    if df[TARGET].nunique() < 2:
        raise ValueError("Target harus memiliki minimal dua kelas")
    return df[FEATURE_COLUMNS + [TARGET]].drop_duplicates().reset_index(drop=True)


def main() -> None:
    source = pd.read_csv(DATA_FILE)
    data = load_and_validate()
    train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data[TARGET])
    data.to_csv(BASE_DIR / "student_ml_dataset.csv", index=False)
    train.to_csv(BASE_DIR / "student_train.csv", index=False)
    test.to_csv(BASE_DIR / "student_test.csv", index=False)
    report = {
        "source_file": DATA_FILE.name,
        "rows": int(len(data)),
        "feature_count": len(FEATURE_COLUMNS),
        "train_rows": len(train),
        "test_rows": len(test),
        "target_distribution": data[TARGET].value_counts().to_dict(),
        "missing_by_column": data.isna().sum().to_dict(),
        "duplicate_rows_removed": int(source.duplicated().sum()),
    }
    (BASE_DIR / "data_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Data siap: {len(train):,} train, {len(test):,} test")
    print("Dibuat: student_ml_dataset.csv, student_train.csv, student_test.csv, data_quality_report.json")


if __name__ == "__main__":
    main()
