import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. Load Dataset
# =========================
DATA_PATH = "Updated_Heart_Disease_Enhanced_Dataset.xlsx"

df = pd.read_excel(DATA_PATH)

print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())


# =========================
# 2. Encode Target
# =========================
# Expected values: 'Absence', 'Presence'
df["Heart Disease"] = df["Heart Disease"].map({
    "Absence": 0,
    "Presence": 1
})

if df["Heart Disease"].isnull().sum() > 0:
    raise ValueError("Target column contains unexpected values. Check 'Heart Disease' labels.")


# =========================
# 3. Define Features and Target
# =========================
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

feature_names = X.columns.tolist()


# =========================
# 4. Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 5. Scale Features
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 6. Train Model
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# =========================
# 7. Evaluate Model
# =========================
y_pred = model.predict(X_test_scaled)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# =========================
# 8. Save Model Assets
# =========================
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\nModel, scaler, and feature names saved successfully.")