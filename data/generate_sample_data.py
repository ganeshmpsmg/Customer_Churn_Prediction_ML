"""
generate_sample_data.py
-----------------------
Generates a realistic, synthetic customer-churn dataset that mirrors the
schema of the IBM Telco Customer Churn dataset. Used as a drop-in sample
so the entire pipeline runs out-of-the-box without needing an external
download. To use the real IBM dataset instead, download it from Kaggle
("blastchar/telco-customer-churn") and place it at
data/raw/telco_churn.csv with the same column names.

Run:
    python data/generate_sample_data.py --n 7043 --out data/raw/telco_churn.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate(n=7043, seed=42):
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, n)

    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], n)
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def dependent_service(base_p_yes=0.5):
        out = rng.choice(["Yes", "No"], n, p=[base_p_yes, 1 - base_p_yes])
        out = np.where(internet_service == "No", "No internet service", out)
        return out

    online_security = dependent_service(0.35)
    online_backup = dependent_service(0.40)
    device_protection = dependent_service(0.40)
    tech_support = dependent_service(0.35)
    streaming_tv = dependent_service(0.45)
    streaming_movies = dependent_service(0.45)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Monthly charges depend loosely on services selected (adds realism)
    base = 18.0
    base += (internet_service == "DSL") * rng.normal(25, 3, n)
    base += (internet_service == "Fiber optic") * rng.normal(55, 5, n)
    base += (online_security == "Yes") * rng.normal(5, 1, n)
    base += (online_backup == "Yes") * rng.normal(5, 1, n)
    base += (device_protection == "Yes") * rng.normal(5, 1, n)
    base += (tech_support == "Yes") * rng.normal(5, 1, n)
    base += (streaming_tv == "Yes") * rng.normal(9, 1, n)
    base += (streaming_movies == "Yes") * rng.normal(9, 1, n)
    base += (phone_service == "Yes") * rng.normal(5, 1, n)
    monthly_charges = np.clip(base, 18, 120).round(2)

    total_charges = (monthly_charges * tenure + rng.normal(0, 10, n)).round(2)
    total_charges = np.clip(total_charges, 0, None)

    # Churn probability driven by known real-world churn drivers
    logit = (
        -1.6
        + 1.4 * (contract == "Month-to-month")
        + 0.35 * (contract == "One year")
        + 0.9 * (internet_service == "Fiber optic")
        + 0.6 * (payment_method == "Electronic check")
        + 0.5 * (paperless_billing == "Yes")
        - 0.03 * tenure
        + 0.015 * (monthly_charges - 65)
        - 0.5 * (tech_support == "Yes")
        - 0.4 * (online_security == "Yes")
        + 0.4 * senior
        - 0.3 * (partner == "Yes")
        - 0.3 * (dependents == "Yes")
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)
    churn_label = np.where(churn == 1, "Yes", "No")

    customer_id = [f"CUST-{100000 + i}" for i in range(n)]

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_label,
    })

    # Inject a few missing values / duplicates to make cleaning realistic
    miss_idx = rng.choice(n, size=int(n * 0.006), replace=False)
    df.loc[miss_idx, "TotalCharges"] = np.nan
    dup_rows = df.sample(5, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7043)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/telco_churn.csv")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
    print(df["Churn"].value_counts(normalize=True))
