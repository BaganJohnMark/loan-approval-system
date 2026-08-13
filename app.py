from flask import Flask, render_template, request, redirect, url_for, session, jsonify, make_response
from functools import wraps
import joblib
import numpy as np
import json
import os
import uuid
import csv
import io
from database import init_db, migrate_db, save_application, get_all_applications, set_verified

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-key-later"

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "adminkami"


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


model = joblib.load("models/best_model.pkl")
all_models = {
    "Logistic Regression": joblib.load("models/model_logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/model_decision_tree.pkl"),
    "Random Forest": joblib.load("models/model_random_forest.pkl"),
    "XGBoost": joblib.load("models/model_xgboost.pkl"),
}
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


def generate_improvement_path(data, prediction):
    if prediction != "Rejected":
        return []
    suggestions = []
    cibil = data["cibil_score"]
    income = data["income_annum"]
    loan = data["loan_amount"]
    total_assets = (
        data["residential_assets_value"] + data["commercial_assets_value"]
        + data["luxury_assets_value"] + data["bank_asset_value"]
    )
    if cibil < 700:
        target_cibil = min(cibil + 150, 750)
        suggestions.append(f"Palakasin ang CIBIL score papuntang {target_cibil}+ sa pamamagitan ng regular at on-time na pagbabayad ng utang sa loob ng 6-12 buwan.")
    if income > 0 and loan > income * 6:
        lower_loan = round(income * 5, -4)
        suggestions.append(f"Bawasan ang hiniram na halaga papuntang mas malapit sa ₱{lower_loan:,.0f} (mas makatwiran kumpara sa kasalukuyang income).")
    if income > 0 and total_assets < income * 1:
        suggestions.append("Magdagdag ng collateral/assets (halimbawa bank savings o property) bilang karagdagang backup sa aplikasyon.")
    if data["loan_term"] <= 4:
        suggestions.append("Pahabain ang loan term (halimbawa 8-10 taon sa halip na mas maikli) para mas mababa ang buwanang bayad at risk.")
    if not suggestions:
        suggestions.append("Panatilihin ang stable na income at magpatuloy sa magandang financial history bago muling mag-apply.")
    return suggestions


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
    input_row = [input_dict[col] for col in feature_columns]
    input_scaled = scaler.transform([input_row])
    pred_encoded = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]
    prediction_label = le_status.inverse_transform([pred_encoded])[0]
    approved_index = list(le_status.classes_).index("Approved")
    approval_probability = float(pred_proba[approved_index])
    risk_tier = get_risk_tier(approval_probability)
    return prediction_label, approval_probability, risk_tier


def get_model_consensus(input_dict):
    input_row = [input_dict[col] for col in feature_columns]
    input_scaled = scaler.transform([input_row])
    approved_index = list(le_status.classes_).index("Approved")
    votes = []
    for name, m in all_models.items():
        pred_encoded = m.predict(input_scaled)[0]
        pred_proba = m.predict_proba(input_scaled)[0]
        label = le_status.inverse_transform([pred_encoded])[0]
        prob = round(float(pred_proba[approved_index]) * 100, 1)
        votes.append({"model": name, "prediction": label, "probability": prob})
    approved_count = sum(1 for v in votes if v["prediction"] == "Approved")
    if approved_count == 4 or approved_count == 0:
        agreement = "High Confidence — Lahat ng models ay sumasang-ayon"
    elif approved_count == 3 or approved_count == 1:
        agreement = "Moderate Confidence — Karamihan sumasang-ayon"
    else:
        agreement = "Model Disagreement — Recommend Manual Review"
    return votes, agreement


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")


@app.route("/predict", methods=["POST"])
def predict():
    applicant_name = request.form.get("applicant_name")

    try:
        no_of_dependents = int(request.form.get("no_of_dependents"))
        income_annum = int(request.form.get("income_annum"))
        loan_amount = int(request.form.get("loan_amount"))
        cibil_score = int(request.form.get("cibil_score"))
        loan_term = int(request.form.get("loan_term"))
        residential_assets_value = int(request.form.get("residential_assets_value"))
        commercial_assets_value = int(request.form.get("commercial_assets_value"))
        luxury_assets_value = int(request.form.get("luxury_assets_value"))
        bank_asset_value = int(request.form.get("bank_asset_value"))
    except (ValueError, TypeError):
        return render_template("error.html", message="Hindi valid ang isa o higit pang numero na inyong inilagay. Pakisuri at subukan ulit."), 400

    if (income_annum < 0 or loan_amount < 0 or no_of_dependents < 0
            or loan_term < 1 or residential_assets_value < 0 or commercial_assets_value < 0
            or luxury_assets_value < 0 or bank_asset_value < 0):
        return render_template("error.html", message="Hindi maaaring negatibo ang mga numerical na value. Pakisuri at subukan ulit."), 400

    if not (300 <= cibil_score <= 900):
        return render_template("error.html", message="Ang CIBIL score ay dapat nasa pagitan ng 300 at 900."), 400

    if income_annum > 10_000_000_000 or loan_amount > 10_000_000_000:
        return render_template("error.html", message="Sobrang laki ng inilagay na numero. Pakisuri ang income o loan amount."), 400

    education = request.form.get("education")
    self_employed = request.form.get("self_employed")

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
    model_votes, model_agreement = get_model_consensus(input_dict)

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
    improvement_suggestions = generate_improvement_path(input_dict, prediction_label)

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
        improvement_suggestions=improvement_suggestions,
        base_input=json.dumps(input_dict),
        max_income=income_annum,
        model_votes=model_votes,
        model_agreement=model_agreement,
    )


@app.route("/simulate", methods=["POST"])
def simulate():
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
    total_applications = len(applications)
    approved_count = sum(1 for a in applications if a["prediction"] == "Approved")
    approval_rate = round((approved_count / total_applications) * 100, 1) if total_applications else 0
    avg_cibil = round(sum(a["cibil_score"] for a in applications) / total_applications) if total_applications else 0
    flagged_count = sum(1 for a in applications if a["flags"])

    return render_template(
        "records.html",
        applications=applications,
        total_applications=total_applications,
        approval_rate=approval_rate,
        avg_cibil=avg_cibil,
        flagged_count=flagged_count,
    )


@app.route("/export/csv")
@login_required
def export_csv():
    applications = get_all_applications()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Name", "Region", "Province", "City", "ID Type", "ID Number", "Verified",
        "Dependents", "Education", "Self Employed", "Income", "Loan Amount", "Loan Term",
        "CIBIL", "Residential Assets", "Commercial Assets", "Luxury Assets", "Bank Assets",
        "Prediction", "Probability", "Risk Tier", "Flags", "Date"
    ])
    for a in applications:
        writer.writerow([
            a["id"], a["applicant_name"], a["region"], a["province"], a["city"],
            a["id_type"], a["id_number"], "Yes" if a["verified"] else "No",
            a["no_of_dependents"], a["education"], a["self_employed"],
            a["income_annum"], a["loan_amount"], a["loan_term"], a["cibil_score"],
            a["residential_assets_value"], a["commercial_assets_value"],
            a["luxury_assets_value"], a["bank_asset_value"],
            a["prediction"], a["probability"], a["risk_tier"], a["flags"], a["created_at"]
        ])
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=credora_applicants.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


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