import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("data/loan_approval_dataset.csv")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Drop loan_id (not a predictive feature)
df = df.drop("loan_id", axis=1)

# Clean whitespace in text columns too
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Encode categorical columns
le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_status = LabelEncoder()

df["education"] = le_education.fit_transform(df["education"])
df["self_employed"] = le_self_employed.fit_transform(df["self_employed"])
df["loan_status"] = le_status.fit_transform(df["loan_status"])

# Save encoders so Flask app can decode predictions later
os.makedirs("models", exist_ok=True)
joblib.dump(le_education, "models/le_education.pkl")
joblib.dump(le_self_employed, "models/le_self_employed.pkl")
joblib.dump(le_status, "models/le_status.pkl")

# Split features (X) and target (y)
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")

print("Preprocessing done.")
print("Training rows:", X_train.shape[0], "| Testing rows:", X_test.shape[0])
print("Feature columns:", list(X.columns))