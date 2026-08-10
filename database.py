import sqlite3

DB_NAME = "loan_applications.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_name TEXT,
            no_of_dependents INTEGER,
            education TEXT,
            self_employed TEXT,
            income_annum INTEGER,
            loan_amount INTEGER,
            loan_term INTEGER,
            cibil_score INTEGER,
            residential_assets_value INTEGER,
            commercial_assets_value INTEGER,
            luxury_assets_value INTEGER,
            bank_asset_value INTEGER,
            prediction TEXT,
            probability REAL,
            risk_tier TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def migrate_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE applications ADD COLUMN flags TEXT DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists, ignore
    conn.close()


def save_application(data):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO applications (
            applicant_name, no_of_dependents, education, self_employed,
            income_annum, loan_amount, loan_term, cibil_score,
            residential_assets_value, commercial_assets_value,
            luxury_assets_value, bank_asset_value,
            prediction, probability, risk_tier, flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["applicant_name"], data["no_of_dependents"], data["education"],
        data["self_employed"], data["income_annum"], data["loan_amount"],
        data["loan_term"], data["cibil_score"], data["residential_assets_value"],
        data["commercial_assets_value"], data["luxury_assets_value"],
        data["bank_asset_value"], data["prediction"], data["probability"],
        data["risk_tier"], data.get("flags", "")
    ))
    conn.commit()
    conn.close()


def get_all_applications():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM applications ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows