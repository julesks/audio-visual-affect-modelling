# src/train_eval_landmarks.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_PATH = Path("data/processed/face_landmark_features.csv")

META_COLS = {"path", "actor", "emotion_code", "emotion", "n_frames_sampled", "n_frames_with_face"}


def load_data():
    df = pd.read_csv(DATA_PATH)

    y = df["emotion"].astype(str).to_numpy()
    groups = df["actor"].to_numpy()

    feature_cols = [c for c in df.columns if c not in META_COLS]

    mean_cols = [c for c in feature_cols if ("mean" in c.lower())]
    std_cols = [c for c in feature_cols if ("std" in c.lower())]

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    X_mean = df[mean_cols].to_numpy(dtype=np.float32)
    X_std = df[std_cols].to_numpy(dtype=np.float32)

    return (X_all, X_mean, X_std), y, groups


def make_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )


def eval_and_print(title: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {title} ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))
    return acc


def random_split_eval(X, y, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = make_model()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return eval_and_print("Random split (stratified by emotion)", y_te, y_pred)


def actor_wise_eval(X, y, groups, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    model = make_model()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    test_actors = sorted(set(groups[test_idx].tolist()))
    print(f"\nActor-wise split test actors: {test_actors}")

    return eval_and_print("Actor-wise split (no actor overlap)", y_te, y_pred)


def main():
    (X_all, X_mean, X_std), y, groups = load_data()

    print(f"Loaded samples: {len(y)} | unique actors: {len(set(groups))}")
    print(f"All features: {X_all.shape[1]} | Mean-only: {X_mean.shape[1]} | Std-only: {X_std.shape[1]}")

    print("\n=== Using ALL features (mean + std) ===")
    acc_rand_all = random_split_eval(X_all, y)
    acc_act_all = actor_wise_eval(X_all, y, groups)

    print("\n=== Using MEAN-only features ===")
    acc_rand_mean = random_split_eval(X_mean, y)
    acc_act_mean = actor_wise_eval(X_mean, y, groups)

    print("\n=== Using STD-only features ===")
    acc_rand_std = random_split_eval(X_std, y)
    acc_act_std = actor_wise_eval(X_std, y, groups)

    print("\n=== Summary ===")
    print(f"All features | random: {acc_rand_all:.4f}  actor-wise: {acc_act_all:.4f}  drop: {(acc_rand_all - acc_act_all):.4f}")
    print(f"Mean only    | random: {acc_rand_mean:.4f}  actor-wise: {acc_act_mean:.4f}  drop: {(acc_rand_mean - acc_act_mean):.4f}")
    print(f"Std only     | random: {acc_rand_std:.4f}  actor-wise: {acc_act_std:.4f}  drop: {(acc_rand_std - acc_act_std):.4f}")


if __name__ == "__main__":
    main()