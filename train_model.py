"""
train_model.py
----------------
Loads data/landmarks.csv (created by collect_data.py), trains a classifier
to predict the sign label from hand-landmark coordinates, and saves the
trained model + label encoder to model.pkl.

HOW TO USE:
    python train_model.py

Requires at least 2 distinct labels with a reasonable number of samples each.
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join("data", "landmarks.csv")
MODEL_PATH = "model.pkl"


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run collect_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} labels: "
          f"{sorted(df['label'].unique())}")

    if df['label'].nunique() < 2:
        print("ERROR: Need at least 2 different labels to train a classifier.")
        return

    X = df.drop("label", axis=1).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "encoder": encoder}, f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
