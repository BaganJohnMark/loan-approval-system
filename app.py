from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from functools import wraps
import joblib
import numpy as np
import json
import os
import uuid
from database import init_db, migrate_db, save_application, get_all_applications, set_verified

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-key-later"

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


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


init_db()
migrate_db()


def get_risk_tier(probability):
    if probability >= 0.80:
        return "Low Risk"
    elif probability >= 0.50:
        return "Medium Risk"
    else:
        return "High Risk"


def check_consistency_flags(data):
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


def generate_rejection_reasons(data, prediction, probability):
    if prediction != "Rejected":
        return []
    reasons = []
    cibil = data["cibil_score"]
    income = data["income_annum"]
    loan = data["loan_amount"]
    total_assets = (
        data["residential_assets_value"] + data["commercial_assets_value"]
        + data["luxury_assets_value"] + data["bank_asset_value"]
    )
    if cibil < 550:
        reasons.append(f"Napakababa ng CIBIL/credit score ({cibil}) — malaking senyales ng mahinang kasaysayan sa pagbabayad ng utang. Ito ang pinaka-malaking factor (80%+) sa desisyon.")
    elif cibil < 700:
        reasons.append(f"Katamtaman lang ang CIBIL score ({cibil}) — hindi ito sapat na kumpiyansa para sa laki ng hiniling na loan.")
    if income > 0 and loan > income * 8:
        ratio = round(loan / income, 1)
        reasons.append(f"Ang hiniling na loan amount ay {ratio}x ng annual income — itinuturing na sobrang laki kumpara sa kakayahang magbayad.")
    if income > 0 and total_assets < income * 0.5:
        reasons.append("Kulang ang assets/collateral kumpara sa income — walang sapat na backup kung sakaling hindi makabayad.")
    if data["loan_term"] <= 4 and loan > income * 4:
        reasons.append("Maikli ang loan term kumbinado sa malaking loan amount — nagreresulta sa mataas na buwanang bayad.")
    if not reasons:
        reasons.append(f"Batay sa kabuuang financial profile, {round(probability*100,1)}% lang ang approval probability — mas mataas ang overall risk kaysa sa itinakdang threshold.")
    return reasons


def run_prediction(input_dict):
    """Core prediction logic — shared by /predict and /simulate."""
    input_row = [input_dict[col] for col in feature_columns]
    input_scaled = scaler.transform([input_row])

    pred_encoded = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]

    prediction_label = le_status.inverse_transform([pred_encoded])[0]
    approved_index = list(le_status.classes_).index("Approved")
    approval_probability = float(pred_proba[approved_index])
    risk_tier = get_risk_tier(approval_probability)

    return prediction_label, approval_probability, risk_tier


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")


@app.route("/predict", methods=["POST"])
def predict():
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

    region = request.form.get("region", "")
    province = request.form.get("province", "")
    city = request.form.get("city", "")
    id_type = request.form.get("id_type", "")
    id_number = request.form.get("id_number", "")

    id_photo_path = ""
    file = request.files.get("id_photo")
    if file and file.filename and allowed_file(file.filename):
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)
        id_photo_path = f"uploads/{unique_name}"

    education_encoded = int(le_education.transform([education])[0])
    self_employed_encoded = int(le_self_employed.transform([self_employed])[0])

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

    prediction_label, approval_probability, risk_tier = run_prediction(input_dict)

    feature_importance = load_feature_importance()
    top_factors = []
    if feature_importance:
        for item in feature_importance[:3]:
            feat = item["feature"]
            top_factors.append({
                "feature": feat.replace("_", " ").title(),
                "value": input_dict[feat],
            })

    flags = check_consistency_flags(input_dict)
    rejection_reasons = generate_rejection_reasons(input_dict, prediction_label, approval_probability)

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
        "region": region,
        "province": province,
        "city": city,
        "id_type": id_type,
        "id_number": id_number,
        "id_photo": id_photo_path,
    })

    return render_template(
        "result.html",
        applicant_name=applicant_name,
        prediction=prediction_label,
        probability=round(approval_probability * 100, 2),
        risk_tier=risk_tier,
        top_factors=top_factors,
        flags=flags,
        rejection_reasons=rejection_reasons,
        base_input=json.dumps(input_dict),
        max_income=income_annum,
    )


@app.route("/simulate", methods=["POST"])
def simulate():
    """Re-run prediction with modified loan_amount / cibil_score for the What-If simulator."""
    data = request.get_json()
    try:
        input_dict = {
            "no_of_dependents": int(data["no_of_dependents"]),
            "education": int(data["education"]),
            "self_employed": int(data["self_employed"]),
            "income_annum": int(data["income_annum"]),
            "loan_amount": int(data["loan_amount"]),
            "loan_term": int(data["loan_term"]),
            "cibil_score": int(data["cibil_score"]),
            "residential_assets_value": int(data["residential_assets_value"]),
            "commercial_assets_value": int(data["commercial_assets_value"]),
            "luxury_assets_value": int(data["luxury_assets_value"]),
            "bank_asset_value": int(data["bank_asset_value"]),
        }
        prediction_label, approval_probability, risk_tier = run_prediction(input_dict)
        return jsonify({
            "prediction": prediction_label,
            "probability": round(approval_probability * 100, 2),
            "risk_tier": risk_tier,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


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


@app.route("/verify/<int:app_id>", methods=["POST"])
@login_required
def verify(app_id):
    set_verified(app_id, 1)
    return redirect(url_for("records"))


@app.route("/insights", methods=["GET"])
@login_required
def insights():
    feature_importance = load_feature_importance()
    return render_template("insights.html", feature_importance=feature_importance)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)