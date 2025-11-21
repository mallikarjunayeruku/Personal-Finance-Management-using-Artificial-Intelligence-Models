import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from pathlib import Path
import json

# ====== CONFIGURATION ======
TRANSACTIONS_PATH = Path("/Users/arrju/Documents/CS595/txn-category-ml/data/expenses_data.csv")  # Replace with actual path
CATEGORIES_PATH = Path("/Users/arrju/Documents/CS595/txn-category-ml/data/categories.json")

# ====== LOAD DATA ======
def load_transactions():
    df = pd.read_csv(TRANSACTIONS_PATH)

    # Parse dates robustly: normalize mixed offsets to UTC, then drop tz for period ops
    dt = pd.to_datetime(df["transactionDate"], errors="coerce", utc=True)
    bad = dt.isna().sum()
    if bad:
        print(f"[WARN] Dropping {bad} rows with unparseable dates.")
    df["transactionDate"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)

    # expenses only (exclude income)
    df["is_income"] = df["is_income"].astype(str).str.lower()
    df = df[df["is_income"].isin(["f", "false", "0"])]

    # amount to numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["transactionDate", "amount", "category_id"])
    return df

def load_categories():
    """Load category mapping from categories.json (optional)."""
    with open(CATEGORIES_PATH, "r") as f:
        data = json.load(f)
    cat_map = {}
    for c in data:
        legacy_name = " > ".join(c["legacy_category"])
        cat_map[str(c["legacy_category_id"])] = legacy_name
    return cat_map

def prepare_monthly_summary(df: pd.DataFrame):
    # Ensure normalized datetimes (from your previous fix)
    df["year_month"] = df["transactionDate"].dt.to_period("M")
    # Aggregate at primary bucket to avoid duplicates like multiple "TRAVEL"
    monthly = (
        df.groupby(["year_month", "category_text"], as_index=False)["amount"]
          .sum()
    )
    return monthly

# ====== EVALUATION HELPERS ======
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R^2 with guard for constant y_true
    if np.allclose(y_true, y_true.mean()):
        r2 = np.nan
    else:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
    # MAPE with guard for zeros
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def backtest_category(series: pd.Series):
    """
    Rolling-origin, one-step-ahead CV for a single category.
    series: pd.Series indexed by sorted year_month (Period), values are monthly totals.
    Returns: y_true, y_pred arrays for this category.
    """
    y = series.values.astype(float)
    n = len(y)
    if n < 4:
        return np.array([]), np.array([])

    y_true_all, y_pred_all = [], []
    # start after at least 3 points for a minimal trend fit
    for t in range(3, n):
        # train on [0..t-1], predict t
        X_train = np.arange(t).reshape(-1, 1)
        y_train = y[:t]
        model = LinearRegression().fit(X_train, y_train)
        pred = model.predict([[t]])[0]
        y_true_all.append(y[t])
        y_pred_all.append(pred)

    return np.asarray(y_true_all), np.asarray(y_pred_all)

def evaluate_model(monthly_df: pd.DataFrame):
    """
    Evaluate with rolling-origin CV per category_text, aggregate metrics overall and per-category.
    Returns (overall_metrics: dict, per_cat_metrics: pd.DataFrame)
    """
    # Ensure proper sorting by time
    df_sorted = monthly_df.sort_values(["category_text", "year_month"]).copy()

    all_true, all_pred, rows = [], [], []
    for cat, sub in df_sorted.groupby("category_text"):
        # Ensure chronological order
        sub = sub.sort_values("year_month")
        # Build a clean numeric sequence per month
        series = pd.Series(sub["amount"].values, index=sub["year_month"].values)
        y_t, y_p = backtest_category(series)
        if len(y_t) == 0:
            continue
        all_true.append(y_t); all_pred.append(y_p)

        m = _compute_metrics(y_t, y_p)
        rows.append({"category_text": cat, **m, "n_forecasts": len(y_t)})

    if len(all_true) == 0:
        overall = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
        per_cat = pd.DataFrame(rows)
        return overall, per_cat

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    overall = _compute_metrics(y_true_all, y_pred_all)
    per_cat = pd.DataFrame(rows).sort_values("MAE")
    return overall, per_cat

# ====== FORECASTING ======
def forecast_next_month(monthly_df: pd.DataFrame):
    last_month = monthly_df["year_month"].max()
    next_month = (last_month + 1).strftime("%Y-%m")

    results = []
    for cat_name in monthly_df["category_text"].unique():
        sub = monthly_df[monthly_df["category_text"] == cat_name].sort_values("year_month").copy()
        if len(sub) < 3:
            continue
        sub["t"] = np.arange(len(sub))
        X = sub[["t"]]
        y = sub["amount"].values

        model = LinearRegression().fit(X, y)
        pred = float(model.predict([[len(sub)]])[0])

        results.append({
            "category_text": cat_name,
            "forecast_amount": max(pred, 0.0),
            "forecast_month": next_month,
        })

    return pd.DataFrame(results)

# ====== MAIN EXECUTION ======
def main():
    print("[INFO] Loading transactions and categories...")
    df = load_transactions()
    _ = load_categories()  # optional; kept for compatibility

    print("[INFO] Preparing monthly data...")
    monthly_df = prepare_monthly_summary(df)

    # ===== Evaluation before final forecast =====
    print("[INFO] Running rolling-origin cross-validation...")
    overall_metrics, per_cat_metrics = evaluate_model(monthly_df)
    print("\n--- Model Evaluation (Overall) ---")
    print(f"MAE : {overall_metrics['MAE']:.2f}")
    print(f"RMSE: {overall_metrics['RMSE']:.2f}")
    r2 = overall_metrics['R2']
    print(f"R²  : {('nan' if pd.isna(r2) else f'{r2:.3f}')}")
    mape = overall_metrics['MAPE']
    print(f"MAPE: {('nan' if pd.isna(mape) else f'{mape:.2f}%')}")

    # (Optional) Show top/bottom categories by MAE
    if not per_cat_metrics.empty:
        print("\n--- Best Categories by MAE (Top 5) ---")
        print(per_cat_metrics[["category_text", "MAE", "RMSE", "R2", "MAPE", "n_forecasts"]].head(5).to_string(index=False))
        print("\n--- Worst Categories by MAE (Top 5) ---")
        print(per_cat_metrics[["category_text", "MAE", "RMSE", "R2", "MAPE", "n_forecasts"]].tail(5).to_string(index=False))

    print("\n[INFO] Forecasting next month spend per category...")
    forecast_df = forecast_next_month(monthly_df)

    # Select top 10 expense categories
    top10 = forecast_df.sort_values("forecast_amount", ascending=False).head(10)
    print("\n --- Top 10 Predicted Expense Categories ---")
    for _, r in top10.iterrows():
        print(f"{r['category_text']} — ${r['forecast_amount']:.2f}")

    print("\n[Info] Forecast completed successfully.")
    return top10, overall_metrics, per_cat_metrics

if __name__ == "__main__":
    main()
