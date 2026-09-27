from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from datetime import datetime

from ml_classifier import predict_category

app = Flask(__name__)

DATABASE = "helpdesk.db"


# ---------------- DATABASE ----------------

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_name TEXT NOT NULL,
            issue TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            solution TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ---------------- ISSUE ANALYSIS ----------------

def analyze_issue(issue):

    # ML-based issue classification
    category, confidence = predict_category(issue)

    solutions = {
        "Network":
            "Check your Wi-Fi or network connection, restart the network adapter, verify VPN settings if applicable, and reconnect. If the issue continues, contact IT support.",

        "Access / Login":
            "Verify your username and password, check whether the account is locked, and use the password reset process if required. Contact IT support if access is still unavailable.",

        "Hardware":
            "Check device connections, power status, and peripheral cables. Restart the affected device and check for visible hardware errors. Contact IT support if the problem continues.",

        "Software":
            "Restart the application, check for available updates, and verify that the required software is installed correctly. If the issue continues, contact IT support.",

        "Other":
            "Provide additional details about the issue so that the support team can investigate it and determine the appropriate resolution."
    }

    # Priority detection
    text = issue.lower()

    if any(word in text for word in [
        "urgent",
        "critical",
        "system down",
        "network down",
        "vpn unavailable",
        "cannot work",
        "unable to work"
    ]):
        priority = "High"

    elif any(word in text for word in [
        "problem",
        "issue",
        "error",
        "not working",
        "not connecting"
    ]):
        priority = "Medium"

    else:
        priority = "Low"

    return category, priority, solutions[category]

# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")

# ---------------- ANALYZE ISSUE ----------------

@app.route("/analyze", methods=["POST"])
def analyze():
    employee_name = request.form["employee_name"]
    issue = request.form["issue"]

    category, priority, solution = analyze_issue(issue)
    
    return render_template(
        "index.html",
        employee_name=employee_name,
        issue=issue,
        category=category,
        priority=priority,
        solution=solution,
        analyzed=True
    )


# ---------------- CREATE TICKET ----------------

@app.route("/create_ticket", methods=["POST"])
def create_ticket():

    employee_name = request.form["employee_name"]
    issue = request.form["issue"]
    category = request.form["category"]
    priority = request.form["priority"]
    solution = request.form["solution"]

    conn = get_db()

    conn.execute("""
        INSERT INTO tickets
        (employee_name, issue, category, priority, solution, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        employee_name,
        issue,
        category,
        priority,
        solution,
        "Open",
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))

    conn.commit()

    ticket_id = conn.execute(
        "SELECT last_insert_rowid()"
    ).fetchone()[0]

    conn.close()

    return render_template(
        "ticket_success.html",
        ticket_id=ticket_id,
        category=category,
        priority=priority
    )

# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():

    conn = get_db()

    tickets = conn.execute(
        "SELECT * FROM tickets ORDER BY id DESC"
    ).fetchall()

    total = conn.execute(
        "SELECT COUNT(*) FROM tickets"
    ).fetchone()[0]

    open_tickets = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE status='Open'"
    ).fetchone()[0]

    resolved = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE status='Resolved'"
    ).fetchone()[0]

    high_priority = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE priority='High'"
    ).fetchone()[0]

    conn.close()

    return render_template(
        "dashboard.html",
        tickets=tickets,
        total=total,
        open_tickets=open_tickets,
        resolved=resolved,
        high_priority=high_priority
    )


# ---------------- RESOLVE TICKET ----------------

@app.route("/resolve/<int:ticket_id>")
def resolve_ticket(ticket_id):

    conn = get_db()

    conn.execute(
        "UPDATE tickets SET status='Resolved' WHERE id=?",
        (ticket_id,)
    )

    conn.commit()
    conn.close()

    return redirect(url_for("dashboard"))


# ---------------- START APPLICATION ----------------

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False)