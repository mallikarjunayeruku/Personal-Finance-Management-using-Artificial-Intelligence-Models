import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ======================== CONFIG ========================
TRANSACTIONS_PATH = Path("/Users/arrju/Documents/CS595/txn-category-ml/data/expenses_data.csv")
TOP_N = 10
MIN_POINTS_RIDGE = 6      # use Ridge if too short for GBM
LAGS = [1, 2, 3]
ROLL = 3                  # rolling window for mean/std
CLIP_LO = 0.01            # winsorization per-category
CLIP_HI = 0.99

# ======================== LOAD & CLEAN ========================
def load_transactions():
    df = pd.read_csv(TRANSACTIONS_PATH)

    # Parse dates robustly: normalize mixed offsets to UTC, drop tz
    dt = pd.to_datetime(df["transactionDate"], errors="coerce", utc=True)
    bad = dt.isna().sum()
    if bad:
        print(f"[WARN] Dropping {bad} rows with unparseable dates.")
    df = df[~dt.isna()].copy()
    df["transactionDate"] = dt[~dt.isna()].dt.tz_convert("UTC").dt.tz_localize(None)

    # Expenses only
    df["is_income"] = df["is_income"].astype(str).str.lower()
    df = df[df["is_income"].isin(["f", "false", "0"])]

    # Types
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount", "transactionDate", "category_text"])

    return df

# ======================== AGGREGATION ========================
def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df["year_month"] = df["transactionDate"].dt.to_period("M")
    monthly = (
        df.groupby(["category_text", "year_month"], as_index=False)["amount"]
          .sum()
    )
    # outlier clipping per category (winsorize)
    def clip_grp(g):
        lo = g["amount"].quantile(CLIP_LO)
        hi = g["amount"].quantile(CLIP_HI)
        g["amount"] = g["amount"].clip(lo, hi)
        return g
    monthly = monthly.groupby("category_text", group_keys=False).apply(clip_grp)
    return monthly.sort_values(["category_text", "year_month"])

# ======================== FEATURES ========================
def add_features(g: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling & seasonal features for a single category frame sorted by time."""
    g = g.copy()
    # time index (trend)
    g["t"] = np.arange(len(g))

    # month seasonality from Period -> timestamp (1..12)
    # Create a timestamp (1st of month) to extract month number
    g["month_num"] = g["year_month"].dt.to_timestamp().dt.month

    # lags & rolling
    for k in LAGS:
        g[f"lag_{k}"] = g["amount"].shift(k)
    g["roll_mean_3"] = g["amount"].rolling(ROLL).mean()
    g["roll_std_3"]  = g["amount"].rolling(ROLL).std(ddof=0)

    return g

def build_matrix(monthly: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for cat, sub in monthly.groupby("category_text"):
        sub = add_features(sub)
        parts.append(sub.assign(category_text=cat))
    X = pd.concat(parts, ignore_index=True)
    return X

# ======================== MODEL & EVAL ========================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if not np.allclose(y_true, y_true.mean()) else np.nan
    mape = np.nanmean(np.abs((y_true - y_pred) / np.where(np.abs(y_true)<1e-12, np.nan, y_true))) * 100.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def fit_predict_one_step(series_df: pd.DataFrame):
    """
    Rolling-origin one-step-ahead CV for a single category using an improved model:
    - GradientBoostingRegressor (nonlinear) with lag/rolling/month features
    - Ridge fallback for very short histories
    """
    cols = ["t", "month_num", *[f"lag_{k}" for k in LAGS], "roll_mean_3", "roll_std_3"]
    df = series_df.dropna(subset=[*cols])  # need full feature row
    if len(df) < 4:
        return np.array([]), np.array([])

    y_true_all, y_pred_all = [], []

    if len(df) < MIN_POINTS_RIDGE:
        model = Ridge(alpha=1.0, random_state=42)
    else:
        model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=42
        )

    # rolling one-step ahead
    for i in range(max(4, ROLL+max(LAGS)), len(series_df)):
        train = df.iloc[:i].dropna(subset=cols)
        test  = df.iloc[i:i+1]
        if test.empty or train.empty:
            continue
        Xtr, ytr = train[cols], train["amount"]
        Xte, yte = test[cols], test["amount"]

        model.fit(Xtr, ytr)
        pred = float(model.predict(Xte)[0])
        y_true_all.append(float(yte.values[0]))
        y_pred_all.append(pred)

    return np.asarray(y_true_all), np.asarray(y_pred_all)

def evaluate(monthly: pd.DataFrame):
    all_true, all_pred, rows = [], [], []
    for cat, sub in monthly.groupby("category_text"):
        sub = sub.sort_values("year_month")
        sub_feat = add_features(sub)
        y_t, y_p = fit_predict_one_step(sub_feat)
        if len(y_t)==0:
            continue
        all_true.append(y_t); all_pred.append(y_p)
        m = compute_metrics(y_t, y_p)
        rows.append({"category_text": cat, **m, "n_forecasts": len(y_t)})
    if not all_true:
        overall = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
        per_cat = pd.DataFrame(rows)
        return overall, per_cat
    overall = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    per_cat = pd.DataFrame(rows).sort_values("MAE")
    return overall, per_cat

# ======================== FORECAST NEXT MONTH ========================
def forecast_next_month(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Train per-category model on all available data and predict t+1 using
    lag/rolling/month features synthesized for the next month.
    """
    results = []
    for cat, sub in monthly.groupby("category_text"):
        sub = sub.sort_values("year_month").copy()
        sub = add_features(sub)

        # choose model
        if len(sub.dropna()) < MIN_POINTS_RIDGE:
            model = Ridge(alpha=1.0, random_state=42)
        else:
            model = GradientBoostingRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=3,
                subsample=0.9, random_state=42
            )

        cols = ["t", "month_num", *[f"lag_{k}" for k in LAGS], "roll_mean_3", "roll_std_3"]

        # latest fully-observed row index
        last = sub.iloc[-1:].copy()
        # synthesize next-row features
        next_period = last["year_month"].iloc[0] + 1
        # Build next feature row
        hist = sub.copy()

        # Create a helper to get lagged values
        def lag_val(k):
            row = hist.iloc[-k:]["amount"].values
            return row[0] if len(row)==1 else row[-1] if len(row)>0 else np.nan

        next_row = {
            "t": int(last["t"].values[0] + 1),
            "month_num": int((next_period).to_timestamp().month),
            "roll_mean_3": float(hist["amount"].tail(ROLL).mean()),
            "roll_std_3":  float(hist["amount"].tail(ROLL).std(ddof=0)),
        }
        for k in LAGS:
            next_row[f"lag_{k}"] = float(hist["amount"].iloc[-k]) if len(hist) >= k else np.nan

        # train on all valid feature rows
        train_df = sub.dropna(subset=cols).copy()
        if train_df.empty or any(np.isnan([next_row.get(c, np.nan) for c in cols])):
            # not enough data to forecast
            continue

        model.fit(train_df[cols], train_df["amount"])
        pred = float(model.predict(pd.DataFrame([next_row])[cols])[0])

        results.append({
            "category_text": cat,
            "forecast_month": str(next_period),
            "forecast_amount": max(pred, 0.0),
        })

    return pd.DataFrame(results).sort_values("forecast_amount", ascending=False)

# ======================== MAIN ========================
def main():
    print("[INFO] Loading & preparing data...")
    tx = load_transactions()
    monthly = to_monthly(tx)

    print("[INFO] Evaluating with rolling-origin CV...")
    overall, per_cat = evaluate(monthly)
    print("\n=== Model Evaluation (Overall) ===")
    print(f"MAE : {overall['MAE']:.2f}")
    print(f"RMSE: {overall['RMSE']:.2f}")
    print(f"R²  : {('nan' if pd.isna(overall['R2']) else f'{overall['R2']:.3f}')}")
    print(f"MAPE: {('nan' if pd.isna(overall['MAPE']) else f'{overall['MAPE']:.2f}%')}")

    if not per_cat.empty:
        print("\n--- Best Categories by MAE (Top 5) ---")
        print(per_cat[["category_text","MAE","RMSE","R2","MAPE","n_forecasts"]].head(5).to_string(index=False))
        print("\n--- Worst Categories by MAE (Top 5) ---")
        print(per_cat[["category_text","MAE","RMSE","R2","MAPE","n_forecasts"]].tail(5).to_string(index=False))

    print("\n[INFO] Forecasting next month (single improved model)...")
    fc = forecast_next_month(monthly)

    print("\n📊 === Top 10 Predicted Expense Categories ===")
    for _, r in fc.head(TOP_N).iterrows():
        print(f"{r['category_text']} — {r['forecast_month']} — ${r['forecast_amount']:.2f}")

    return fc, overall, per_cat

if __name__ == "__main__":
    main()
