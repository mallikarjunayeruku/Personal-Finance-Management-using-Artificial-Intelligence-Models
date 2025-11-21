# income_forecast.py
import pandas as pd
import numpy as np
import psycopg2
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# 1. Database Connection
# -----------------------------
conn = psycopg2.connect(
    dbname="financeai",
    user="arjun",
    password="arjun123",
    host="localhost",
    port="5432"
)

query = """
WITH allowed AS (
    SELECT id, name as cat_name, description AS category_text
    FROM finance_category
)
SELECT
    t.id,
    COALESCE(t.amount, 0) AS amount,
    t."transactionDate"
FROM finance_transactions t
JOIN allowed a ON a.id = t.category_id
WHERE t.category_id IS NOT NULL 
  AND t."isIncome" = true
  AND t.user_id = 7
ORDER BY t."transactionDate";
"""

df = pd.read_sql(query, conn)
conn.close()

# -----------------------------
# 2. Preprocessing
# -----------------------------
df["transactionDate"] = pd.to_datetime(df["transactionDate"], utc=True)
df = df.groupby(df["transactionDate"].dt.to_period("M"))["amount"].sum().reset_index()
df["transactionDate"] = df["transactionDate"].dt.to_timestamp()
df = df.set_index("transactionDate")
monthly = df.groupby(df["transactionDate"].dt.to_period("M"))["amount"].sum().to_timestamp().to_frame("income")

# Create lag features
#df["prev1"] = df["amount"].shift(1)
#df["prev2"] = df["amount"].shift(2)
#df["prev3"] = df["amount"].shift(3)
#df = df.dropna()

# -----------------------------
# 3. Model Training
# -----------------------------
#X = df[["prev1", "prev2", "prev3"]]
#y = df["amount"]


# 1️⃣ Add more lag features (increasing model memory)
for k in range(1, 6):  # previously 3 lags → now 5 lags
    monthly[f"prev{k}"] = monthly["income"].shift(k)
monthly = monthly.dropna()

# Features and target
X = monthly[[f"prev{k}" for k in range(1, 6)]]
y = monthly["income"]


#model = LinearRegression()
#tscv = TimeSeriesSplit(n_splits=5)

# 2️⃣ Use Ridge Regression instead of simple LinearRegression
# Ridge introduces alpha (L2 regularization) to prevent overfitting
model = Ridge(alpha=0.5, fit_intercept=True, positive=False)

# 3️⃣ Increase TimeSeriesSplit folds for more robust backtesting
tscv = TimeSeriesSplit(n_splits=8)
mae_scores = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae_scores.append(mean_absolute_error(y_test, y_pred))

avg_mae = np.mean(mae_scores)
print(f"[INFO] Backtest MAE (mean of {len(mae_scores)} folds): {avg_mae:.2f}")

# -----------------------------
# 4. Predict Next Month Income
# -----------------------------
latest = df.iloc[-1]
future_features = [[latest["prev1"], latest["prev2"], latest["prev3"]]]

forecast = model.predict(future_features)[0]
next_month = (df.index[-1] + pd.offsets.MonthBegin(1)).strftime("%Y-%m")

print(f"[RESULT] Forecast for {next_month}: ${forecast:.0f}")

# -----------------------------
# 5. Confidence Band (P50/P90)
# -----------------------------
p50 = np.percentile(y, 50)
p90 = np.percentile(y, 90)
print(f"[RESULT] Budget Band — P50=${p50:.0f}, P90=${p90:.0f}")
