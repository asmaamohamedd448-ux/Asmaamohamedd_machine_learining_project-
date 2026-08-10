"""
Train a diabetes risk classifier and save everything needed for deployment
(model + scaler + encoding maps) into a single model.pkl file.

Design decisions (differ slightly from the original notebook, on purpose):
- Dropped 'location' (50 US states, high-cardinality, near-zero signal) and
  'year' (not a real risk factor) to keep the deployed form short and sensible.
- Dropped the one-hot 'race:*' columns from the model inputs to keep the
  input form simple and avoid using race as a predictive feature.
- Fit the StandardScaler on the TRAINING split only (the original notebook
  scaled before the train/test split, which leaks test info into training).
- Used class_weight='balanced' in RandomForest since diabetes=1 is only
  ~8.5% of the data — this trades a little precision for much better recall
  on the minority (diabetic) class, which matters more in a health screening
  context (missing a diabetic case is worse than a false alarm).
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------- Load ----------
df = pd.read_csv("diabetes_dataset.csv")

# ---------- Clean / encode ----------
gender_map = {"Male": 0, "Female": 1, "Other": 1}  # 'Other' folded into 1 (18 rows, negligible)
smoking_map = {
    "No Info": 0, "never": 1, "former": 2,
    "current": 3, "not current": 4, "ever": 5,
}

df["gender"] = df["gender"].map(gender_map)
df["smoking_history"] = df["smoking_history"].map(smoking_map)

feature_cols = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "hbA1c_level", "blood_glucose_level",
]
target_col = "diabetes"

X = df[feature_cols]
y = df[target_col]

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Scale (fit on train only) ----------
scale_cols = ["age", "bmi", "hbA1c_level", "blood_glucose_level"]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

# ---------- Train ----------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)

# ---------- Evaluate ----------
y_pred = model.predict(X_test_scaled)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1:        {f1_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------- Save everything needed for inference ----------
artifact = {
    "model": model,
    "scaler": scaler,
    "gender_map": gender_map,
    "smoking_map": smoking_map,
    "feature_cols": feature_cols,
    "scale_cols": scale_cols,
}
joblib.dump(artifact, "model.pkl")
print("\nSaved model.pkl")
