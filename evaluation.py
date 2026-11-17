#!/usr/bin/env python3
"""Train classifiers on a CSV dataset and report standard evaluation metrics."""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

NPI_FEATURES = [
    "duration_sec", "total_bytes", "packet_count", "flow_count",
    "up_bytes_client_to_server", "down_bytes_server_to_client",
    "up_packets_client_to_server", "down_packets_server_to_client",
    "tcp_share", "udp_share",
    "packet_size_mean", "packet_size_std", "packet_size_min",
    "packet_size_q1", "packet_size_median", "packet_size_q3",
    "packet_size_max", "packet_size_iqr",
    "interarrival_mean", "interarrival_std", "interarrival_min",
    "interarrival_q1", "interarrival_median", "interarrival_q3",
    "interarrival_max", "interarrival_iqr",
    "payload_len_mean", "payload_len_std", "payload_len_min",
    "payload_len_q1", "payload_len_median", "payload_len_q3",
    "payload_len_max", "payload_len_iqr",
    "ttl_mean", "ttl_std", "ttl_min",
    "ttl_q1", "ttl_median", "ttl_q3",
    "ttl_max", "ttl_iqr",
    "packets_per_second", "bits_per_second",
    "up_down_packet_ratio", "up_down_byte_ratio",
    "burstiness_cv_interarrival", "size_variability_cv",
]

MODELS = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42, max_iter=500
    ),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": GaussianNB(),
}


def main():
    csv_url = input("Paste the CSV file URL: ").strip()
    if not csv_url:
        raise ValueError("A CSV file URL is required.")

    df = pd.read_csv(csv_url)
    if "label" not in df.columns:
        raise ValueError("The CSV must contain a 'label' column.")

    missing_features = [column for column in NPI_FEATURES if column not in df.columns]
    if missing_features:
        raise ValueError(
            "The CSV is missing required feature columns: "
            + ", ".join(missing_features)
        )

    X = df[NPI_FEATURES].copy()
    y = df["label"].copy()

    labels = set(y.dropna().unique())
    if labels != {0, 1}:
        raise ValueError(
            "The 'label' column must contain both binary classes 0 and 1."
        )

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Algorithm": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1-score": f1_score(y_test, predictions, zero_division=0),
                "AU-ROC": roc_auc_score(y_test, probabilities),
            }
        )

    results_df = pd.DataFrame(results)
    for metric in ("Accuracy", "Precision", "Recall", "F1-score", "AU-ROC"):
        results_df[metric] = results_df[metric].map(lambda score: f"{score:.4f}")

    print(f"\nEvaluation results ({len(X_train)} training, {len(X_test)} test samples):")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
