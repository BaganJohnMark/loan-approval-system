from flask import Flask, render_template, request, redirect, url_for, session
from functools import wraps
import joblib
import numpy as np
import json
from database import init_db, migrate_db, save_application, get_all_applications

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-key-later"

# ----- Admin credentials (simple version) -----
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ----- Login-required decorator -----
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# ----- Load trained model and preprocessing objects -----
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le_education = joblib.load("models/le_education.pkl")
le_self_employed = joblib.load("models/le_self_employed.pkl")
le_status = joblib.load("models/le_status.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


def load_feature_importance():
    try:
        with open("models/feature_importance.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Initialize database (creates table if it doesn't exist yet)
init_db()
migrate_db()


def get_risk_tier(probability):
    """Classify risk tier based on model's approval probability."""
    if probability >= 0.80:
        return "Low Risk"
    elif probability >= 0.50:
        return "Medium Risk"
    else:
        return "High Risk"


def check_consistency_flags(data):
    """Rule-based checks for suspicious/inconsistent applicant data."""
    flags = []
    income = data["income_annum"]
    loan = data["loan_amount"]
    total_assets = (
        data["residential_assets_value"] + data["commercial_assets_value"]
        + data["luxury_assets_value"] + data["bank_asset_value"]
    )

    if income > 0 and loan > income * 10:
        flags.append("Loan amount unusually high vs. income (>10x)")

    if income > 0 and total_assets > income * 30:
        flags.append("Asset values unusually high vs. income (>30x)")

    if income > 0 and total_assets < income * 0.05:
        flags.append("Very low assets relative to income (possible missing/false asset info)")

    if data["no_of_dependents"] > 10:
        flags.append("Unusually high number of dependents")

    return flags


@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ----- Get form data -----
    applicant_name = request.form.get("applicant_name")
    no_of_dependents = int(request.form.get("no_of_dependents"))
    education = request.form.get("education")
    self_employed = request.form.get("self_employed")
    income_annum = int(request.form.get("income_annum"))
    loan_amount = int(request.form.get("loan_amount"))
    loan_term = int(request.form.get("loan_term"))
    cibil_score = int(request.form.get("cibil_score"))
    residential_assets_value = int(request.form.get("residential_assets_value"))
    commercial_assets_value = int(request.form.get("commercial_assets_value"))
    luxury_assets_value = int(request.form.get("luxury_assets_value"))
    bank_asset_value = int(request.form.get("bank_asset_value"))

    # ----- Encode categorical fields the same way as training -----
    education_encoded = le_education.transform([education])[0]
    self_employed_encoded = le_self_employed.transform([self_employed])[0]

    # ----- Build feature vector in the exact same column order as training -----
    input_dict = {
        "no_of_dependents": no_of_dependents,
        "education": education_encoded,
        "self_employed": self_employed_encoded,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }
    input_row = [input_dict[col] for col in feature_columns]
    input_scaled = scaler.transform([input_row])

    # ----- Predict -----
    pred_encoded = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]

    prediction_label = le_status.inverse_transform([pred_encoded])[0]
    # Probability of the "Approved" class specifically
    approved_index = list(le_status.classes_).index("Approved")
    approval_probability = float(pred_proba[approved_index])
    risk_tier = get_risk_tier(approval_probability)

    # ----- Build top contributing factors for this applicant -----
    feature_importance = load_feature_importance()
    top_factors = []
    if feature_importance:
        for item in feature_importance[:3]:
            feat = item["feature"]
            top_factors.append({
                "feature": feat.replace("_", " ").title(),
                "value": input_dict[feat],
            })

    # ----- Consistency / suspicious-data checks -----
    flags = check_consistency_flags(input_dict)

    # ----- Save to database (the "sheet") -----
    save_application({
        "applicant_name": applicant_name,
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
        "prediction": prediction_label,
        "probability": round(approval_probability * 100, 2),
        "risk_tier": risk_tier,
        "flags": "; ".join(flags),
    })

    return render_template(
        "result.html",
        applicant_name=applicant_name,
        prediction=prediction_label,
        probability=round(approval_probability * 100, 2),
        risk_tier=risk_tier,
        top_factors=top_factors,
        flags=flags,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("records"))
        else:
            error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))


@app.route("/records", methods=["GET"])
@login_required
def records():
    applications = get_all_applications()
    return render_template("records.html", applications=applications)


@app.route("/insights", methods=["GET"])
@login_required
def insights():
    feature_importance = load_feature_importance()
    return render_template("insights.html", feature_importance=feature_importance)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)