"""
monitoring.py
--------------
Bonus feature: lightweight real-time churn monitoring + email alerting.

- `scan_for_high_risk` scores a batch of customers and returns anyone above
  an alert threshold.
- `send_email_alert` sends an SMTP email (configure via environment
  variables) summarizing newly-flagged high-risk customers. This is a
  best-effort integration meant to be wired into a scheduler (cron,
  Airflow, GitHub Actions schedule, etc.) — it will not run automatically.

Environment variables (all optional; alerting is skipped if unset):
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO

Usage:
    python src/monitoring.py --input data/raw/telco_churn.csv --threshold 0.8
"""
import argparse
import os
import smtplib
import sys
from email.mime.text import MIMEText

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ChurnPredictor
from utils import get_logger, REPORTS_DIR

logger = get_logger(__name__)


def scan_for_high_risk(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    predictor = ChurnPredictor()
    results = predictor.predict_batch(df, explain=False)
    flagged = results[results["churn_probability"] >= threshold].sort_values(
        "churn_probability", ascending=False
    )
    logger.info(f"Scanned {len(df)} customers; {len(flagged)} exceed the "
                f"{threshold:.0%} churn-probability alert threshold.")
    return flagged


def send_email_alert(flagged_df: pd.DataFrame) -> bool:
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to_addr = os.environ.get("ALERT_EMAIL_TO")

    if not all([host, user, password, to_addr]):
        logger.warning("SMTP environment variables not fully configured; skipping email send. "
                        "Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO to enable alerts.")
        return False

    body_lines = [f"{len(flagged_df)} customers exceeded the high-risk churn threshold:\n"]
    for _, row in flagged_df.head(20).iterrows():
        body_lines.append(
            f"- {row['customer_id']}: {row['churn_probability']*100:.1f}% churn probability "
            f"(${row['estimated_annual_revenue_at_risk']:.0f}/yr at risk)"
        )
    body = "\n".join(body_lines)

    msg = MIMEText(body)
    msg["Subject"] = f"[Churn Alert] {len(flagged_df)} high-risk customers detected"
    msg["From"] = user
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
        logger.info(f"Alert email sent to {to_addr}.")
        return True
    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")
        return False


def main(input_path: str, threshold: float):
    df = pd.read_csv(input_path)
    flagged = scan_for_high_risk(df, threshold)

    out_path = os.path.join(REPORTS_DIR, "high_risk_alerts.csv")
    flagged.to_csv(out_path, index=False)
    logger.info(f"Saved flagged customers to {out_path}")

    if len(flagged) > 0:
        send_email_alert(flagged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    main(args.input, args.threshold)
