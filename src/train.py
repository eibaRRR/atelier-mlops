import os
import yaml
import json
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

with open("params.yaml", "r") as file:
    all_params = yaml.safe_load(file)
params = all_params["train"]
knn_params = all_params["knn"]

df = pd.read_csv("data/raw/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"],
    random_state=params["random_state"], stratify=y
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted"),
        "recall": recall_score(y_test, predictions, average="weighted"),
        "f1_score": f1_score(y_test, predictions, average="weighted"),
    }


# --- Modele 1 : Random Forest ---
with mlflow.start_run(run_name="RandomForest_Iris"):
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    joblib.dump(model, "models/model.pkl")
    mlflow.log_artifact("models/model.pkl", artifact_path="model")

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("RandomForest - Metriques :", metrics)

# --- Modele 2 : KNN (TP3 : comparaison de deux modeles) ---
with mlflow.start_run(run_name="KNN_Iris"):
    knn = KNeighborsClassifier(
        n_neighbors=knn_params["n_neighbors"],
        weights=knn_params["weights"]
    )
    knn.fit(X_train, y_train)
    knn_metrics = evaluate(knn, X_test, y_test)

    mlflow.log_params(knn_params)
    mlflow.log_metrics(knn_metrics)
    joblib.dump(knn, "models/knn_model.pkl")
    mlflow.log_artifact("models/knn_model.pkl", artifact_path="model")

    with open("reports/knn_metrics.json", "w") as f:
        json.dump(knn_metrics, f, indent=4)

    print("KNN - Metriques :", knn_metrics)

print("Modeles entraines et enregistres dans MLflow.")
