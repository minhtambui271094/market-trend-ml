import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("data/mtf_dataset.csv")
df = df.sort_values("time").reset_index(drop=True)

print("Dataset Shape:", df.shape)

print("\nTarget Distribution:")
print(df["target"].value_counts())


# ==========================
# CLEAN DATA
# ==========================

df = df.dropna()

features = [c for c in df.columns if c not in ["time", "target"]]

X = df[features]
y = df["target"]


# ==========================
# WALK-FORWARD CONFIG
# ==========================

n_splits = 5
step = len(df) // 10
window = len(df) // 3

fold_scores = []


# ==========================
# WALK-FORWARD TRAINING
# ==========================

for i in range(n_splits):

    start = i * step
    train_end = start + window
    test_end = train_end + step

    X_train = X.iloc[start:train_end]
    y_train = y.iloc[start:train_end]

    X_test = X.iloc[train_end:test_end]
    y_test = y.iloc[train_end:test_end]

    print("\n=========================")
    print(f"FOLD {i+1}")
    print("=========================")

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    print("\nTest class distribution:")
    print(y_test.value_counts())


    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("\nAccuracy:", round(acc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    fold_scores.append(acc)


# ==========================
# FINAL RESULT
# ==========================

print("\n=========================")
print("WALK-FORWARD RESULT")
print("=========================")

print("Mean Accuracy:", round(np.mean(fold_scores), 4))
print("Std Accuracy :", round(np.std(fold_scores), 4))


# ==========================
# TRAIN FINAL MODEL
# ==========================

final_model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="mlogloss"
)

final_model.fit(X, y)

joblib.dump(final_model, "models/trend_model.pkl")

print("\nModel Saved -> models/trend_model.pkl")