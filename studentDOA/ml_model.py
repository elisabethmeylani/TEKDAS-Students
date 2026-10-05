"""Train and evaluate a multiclass student outcome model."""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import BASE_DIR, CATEGORICAL_COLUMNS, FEATURES_BY_CUTOFF, TARGET

def build_pipeline(feature_columns) -> Pipeline:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    preprocess = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    classifier = RandomForestClassifier(n_estimators=220, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1)
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def train_model() -> Pipeline:
    train = pd.read_csv(BASE_DIR / "student_train.csv")
    test = pd.read_csv(BASE_DIR / "student_test.csv")
    all_metrics = {}
    for cutoff, features in FEATURES_BY_CUTOFF.items():
        model = build_pipeline(features)
        model.fit(train[features], train[TARGET])
        predictions = model.predict(test[features])
        all_metrics[cutoff] = {
            "accuracy": float(accuracy_score(test[TARGET], predictions)),
            "macro_f1": float(f1_score(test[TARGET], predictions, average="macro")),
            "weighted_f1": float(f1_score(test[TARGET], predictions, average="weighted")),
            "labels": sorted(test[TARGET].unique().tolist()),
            "confusion_matrix": confusion_matrix(test[TARGET], predictions, labels=model.classes_).tolist(),
            "classification_report": classification_report(test[TARGET], predictions, output_dict=True, zero_division=0),
        }
        suffix = cutoff.lower().replace(" ", "")
        joblib.dump(model, BASE_DIR / f"student_outcome_{suffix}.pkl")
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).to_csv(BASE_DIR / f"feature_importance_{suffix}.csv", index=False)
        scored = test[[TARGET]].copy()
        scored["predicted_status"] = predictions
        probabilities = model.predict_proba(test[features])
        for i, label in enumerate(model.classes_):
            scored[f"probability_{label}"] = probabilities[:, i]
        scored.to_csv(BASE_DIR / f"student_predictions_{suffix}.csv", index=False)
        print(f"=== Model {cutoff} ===")
        print(classification_report(test[TARGET], predictions, zero_division=0))
    (BASE_DIR / "model_metrics.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Model Semester 1 dan Semester 2 berhasil disimpan.")
    return model


if __name__ == "__main__":
    train_model()
