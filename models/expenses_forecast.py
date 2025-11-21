# expenses_forecast.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV, QuantileRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# ==== 1) Load transactions safely ============================================
DATA_PATH = "/Users/arrju/Documents/CS595/txn-category-ml/data/expenses_data.csv"
CAT_DATA_PATH = "/Users/arrju/Documents/CS595/txn-category-ml/data/categories.json"

# Read CSV and detect a date column robustly
df = pd.read_csv(DATA_PATH)

# Try to find a date column
DATE_COLS = [
    "transactionDate", "transaction_date", "date",
    "createdDate", "created_at", "posted_at"
]
date_col = next((c for c in DATE_COLS if c in df.columns), None)
if date_col is None:
    raise ValueError(f"No date column found in file. Tried: {DATE_COLS}")

# Parse and clean date
df["transactionDate"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
df = df.dropna(subset=["transactionDate"])

# Coerce amount column safely
if "amount" not in df.columns:
    raise ValueError("Missing required 'amount' column in data file.")
df["amount"] = (
    df["amount"]
    .astype(str)
    .str.replace("[,$]", "", regex=True)
    .str.replace(r"\((.*)\)", r"-\1", regex=True)  # handle (123) negatives
    .str.replace("\u2212", "-", regex=False)
    .astype(float)
)

# ==== 2) Filter out income & transfers =======================================
# Option A: if you already have category names (cat_name)
if "cat_name" in df.columns:
    keep = ~df["cat_name"].isin(["INCOME", "TRANSFER_IN", "TRANSFER_OUT"])
else:
    # Option B: join from categories.json if available
    try:
        cat_map = pd.json_normalize(json.loads(Path(CAT_DATA_PATH).read_text()))
        if "id" in cat_map.columns and "possible_pfcs" in cat_map.columns and "category_id" in df.columns:
            cat_map["primary"] = cat_map["possible_pfcs"].apply(
                lambda x: x[0].get("primary") if isinstance(x, list) and x and isinstance(x[0], dict) else None
            )
            df = df.merge(cat_map[["id", "primary"]], left_on="category_id", right_on="id", how="left")
            keep = ~df["primary"].isin(["INCOME", "TRANSFER_IN", "TRANSFER_OUT"])
        else:
            keep = df["amount"].notna()
    except Exception as e:
        print(f"[WARN] Could not parse category map: {e}")
        keep = df["amount"].notna()

df = df[keep].copy()
print(f"[INFO] Using {len(df)} filtered transactions for forecasting.")

# ==== 3) Aggregate monthly expenses ==========================================
df["year_month"] = df["transactionDate"].dt.to_period("M").dt.to_timestamp()
monthly = df.groupby("year_month", as_index=False)["amount"].sum().sort_values("year_month")

# ==== 4) Feature engineering (lags, rolling, seasonality) =====================
def add_lags(s, lags):
    return pd.DataFrame({f"lag_{k}": s.shift(k) for k in lags})

def add_rolls(s):
    return pd.DataFrame({
        "roll_mean_3": s.rolling(3).mean(),
        "roll_mean_6": s.rolling(6).mean(),
        "roll_mean_12": s.rolling(12).mean(),
    })

y = monthly["amount"].astype(float)
X_num = pd.concat([add_lags(y, range(1, 13)), add_rolls(y)], axis=1)
X_cat = pd.DataFrame({"month_num": monthly["year_month"].dt.month.astype(int)})

X = pd.concat([X_num, X_cat], axis=1)
data = pd.concat([monthly[["year_month"]], X, y.rename("target")], axis=1).dropna().reset_index(drop=True)

# ==== 5) TimeSeries cross-validation =========================================
if len(data) < 12:
    print(f"[WARN] Only {len(data)} months of data — model may be less accurate.")

tscv = TimeSeriesSplit(n_splits=min(6, max(2, len(data)//4)))
num_cols = [c for c in X.columns if c != "month_num"]
cat_cols = ["month_num"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = RidgeCV(alphas=np.logspace(-3, 3, 21))
pipe = Pipeline([("pre", pre), ("model", model)])

maes = []
for tr_idx, te_idx in tscv.split(data):
    tr, te = data.iloc[tr_idx], data.iloc[te_idx]
    pipe.fit(tr.drop(columns=["year_month", "target"]), tr["target"])
    pred = pipe.predict(te.drop(columns=["year_month", "target"]))
    maes.append(mean_absolute_error(te["target"], pred))

print(f"[INFO] Backtest MAE (mean of {len(maes)} folds): {np.mean(maes):.2f}")

# ==== 6) Refit and forecast next month =======================================
pipe.fit(data.drop(columns=["year_month", "target"]), data["target"])

last_ym = data["year_month"].max()
next_ym = last_ym + pd.offsets.MonthBegin(1)

full_y = monthly.set_index("year_month")["amount"].astype(float)
Xn = pd.concat([add_lags(full_y, range(1, 13)), add_rolls(full_y)], axis=1).iloc[[-1]].copy()
Xn["month_num"] = int(next_ym.month)

next_expense = pipe.predict(Xn)[0]
print(f"[RESULT] Forecast for {next_ym.strftime('%Y-%m')}: ${next_expense:,.0f}")

# ==== 7) Optional: Quantile regression for budget band =======================
try:
    q50 = Pipeline([("pre", pre), ("q", QuantileRegressor(quantile=0.5, alpha=1.0))])
    q90 = Pipeline([("pre", pre), ("q", QuantileRegressor(quantile=0.9, alpha=1.0))])

    q50.fit(data.drop(columns=["year_month", "target"]), data["target"])
    q90.fit(data.drop(columns=["year_month", "target"]), data["target"])

    p50 = q50.predict(Xn)[0]
    p90 = q90.predict(Xn)[0]
    print(f"[RESULT] Budget Band — P50=${p50:,.0f}, P90=${p90:,.0f}")
except Exception as e:
    print(f"[INFO] Quantile regression skipped: {e}")
