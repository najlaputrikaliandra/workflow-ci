import os
import shutil
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from scipy.stats import randint
from sklearn.utils import estimator_html_repr

# MLflow Local Tracking
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

def load_and_split_data(data_path, target_col="target", test_size=0.2):
    print(f"Memuat dataset: {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.select_dtypes(include=["int64", "float64"])

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_names 

def train_with_tuning(X_train, y_train, X_test, y_test, feature_names):

    with mlflow.start_run(run_name="RF_Classifier_Tuning") as run:

        param_dist = {
            "rf__n_estimators": randint(100, 800),
            "rf__max_depth": [None, 10, 20, 30],
            "rf__min_samples_split": randint(2, 20),
            "rf__min_samples_leaf": randint(1, 10),
            "rf__max_features": ["sqrt", "log2"]
        }

        pipeline = Pipeline(steps=[
            ("rf", RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            ))
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=50,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        print("Training + Tuning dimulai...")
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_score", search.best_score_)

        y_pred = best_model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1-score": f1_score(y_test, y_pred, average="weighted")
        }

        mlflow.log_metrics(metrics)
        print("Metrics:", metrics)

        
        with open("metric_info.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        with open("estimator.html", "w", encoding="utf-8") as f:
            f.write(estimator_html_repr(best_model))
        mlflow.log_artifact("estimator.html")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        with open("features.txt", "w") as f:
            for feat in feature_names:
                f.write(f"{feat}\n")
        mlflow.log_artifact("features.txt")

        model_dir = "rf_classifier_model"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        mlflow.sklearn.save_model(best_model, model_dir)
        mlflow.log_artifact(model_dir)

        print(f"\nTraining selesai | Run ID: {run.info.run_id}")

if __name__ == "__main__":
    DATA_PATH = "heart_clean.csv"

    X_train, X_test, y_train, y_test, feature_names = load_and_split_data(DATA_PATH)
    train_with_tuning(X_train, y_train, X_test, y_test, feature_names)