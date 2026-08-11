import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ----- Load and prepare data (same steps as preprocess.py) -----
df = pd.read_csv("data/loan_approval_dataset.csv")
df.columns = df.columns.str.strip()
df = df.drop("loan_id", axis=1)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

le_education = joblib.load("models/le_education.pkl")
le_self_employed = joblib.load("models/le_self_employed.pkl")
le_status = joblib.load("models/le_status.pkl")

df["education"] = le_education.transform(df["education"])
df["self_employed"] = le_self_employed.transform(df["self_employed"])
df["loan_status"] = le_status.transform(df["loan_status"])

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = joblib.load("models/scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Define models -----
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

results = []
best_model = None
best_model_name = None
best_f1 = -1

# ----- Train and evaluate each model -----
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    false_approval_rate = fp / (fp + tn)   # rejected wrongly approved
    false_rejection_rate = fn / (fn + tp)  # approved wrongly rejected

    results.append({
        "Model": name,
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "False Approval Rate": round(false_approval_rate * 100, 2),
        "False Rejection Rate": round(false_rejection_rate * 100, 2),
    })

    print(f"\n{name}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  False Approval Rate:  {false_approval_rate*100:.2f}%")
    print(f"  False Rejection Rate: {false_rejection_rate*100:.2f}%")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# ----- Save results table -----
results_df = pd.DataFrame(results)
results_df.to_csv("models/model_comparison.csv", index=False)
print("\n\nComparison table saved to models/model_comparison.csv")
print(results_df)

# ----- Save the best model -----
joblib.dump(best_model, "models/best_model.pkl")
print(f"\nBest model: {best_model_name} (F1-Score: {best_f1:.4f}) saved to models/best_model.pkl")
# ----- Save feature importance for explainability -----
import json

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = abs(best_model.coef_[0])
else:
    importances = None

if importances is not None:
    feature_importance = sorted(
        zip(X.columns, importances),
        key=lambda x: x[1],
        reverse=True
    )
    feature_importance_list = [
        {"feature": f, "importance": round(float(i), 4)}
        for f, i in feature_importance
    ]
    with open("models/feature_importance.json", "w") as f:
        json.dump(feature_importance_list, f, indent=2)

    print("\nFeature Importance:")
    for f, i in feature_importance:
        print(f"  {f}: {i:.4f}")

    print("\nSaved to models/feature_importance.json")
    # ----- Save ALL models for consensus view -----
model_filenames = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}
for name, model_obj in models.items():
    filename = f"models/model_{model_filenames[name]}.pkl"
    joblib.dump(model_obj, filename)
    print(f"Saved {name} to {filename}")