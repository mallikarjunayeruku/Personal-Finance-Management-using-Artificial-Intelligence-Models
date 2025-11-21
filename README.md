# Personal Finance with AI — Machine Learning Notebooks & Models

This README explains the **machine learning notebooks and Python models** used in the project, how the data flows into them, what each model does, expected results, and **exact steps to run** everything locally or in a notebook runner.

> For an overview of the problem statement, project scope, and expected outcomes, see the **CSUDH CSC 590 project proposal PDF** included in this repo.  
> For Python backend dependencies, see `requirements.txt`. For optional UI/plot dependencies on the web app, see `package.json` / `package-lock.json`.  
> Category mapping reference for transactions is under `categories.json`.

---

## 1) Environment Setup

### Python (backend & notebooks)

- **Python**: 3.10–3.12 recommended
- **Create venv and install deps**
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

- **Start Jupyter**
  ```bash
  jupyter lab    # or: jupyter notebook
  ```

## 2) Data Sources & Configuration

- **Primary store**: PostgreSQL (historical transactions; account-level info).  
- **Secondary/ingestion**: Plaid Sandbox (synthetic accounts + transactions).  
- **Local reference files**:
  - `categories.json` — mapping of Plaid legacy categories to **Primary Financial Categories (PFCs)** used by the models.
- **Environment variables** (typical):
  ```bash
  # database
  export DATABASE_URL=postgresql://user:pass@localhost:5432/finance_db
  ```

**Data columns expected** (common subset across notebooks):
- `id` (txn id), `account_id`, `date` (or `transactionDate`), `amount` (positive = outflow; negative = inflow for non-Income endpoints), `merchant_name`, `name` or `description`, `category_id`/`cat_name` (optional).

> Notes on income/expense sign convention: For Plaid Transactions (non-Income products), **positive** amounts mean **money out**; **negative** mean **money in**. For Income endpoints, amounts are **positive** for income. Align your preprocessing accordingly.

---

## 3) Notebooks & Models

> Folder structure below is a suggestion. If your project uses different paths, adjust the commands accordingly.


### A. Monthly Expense Forecast
The Monthly Expense Forecasting Model is designed to predict a user’s upcoming month’s total spending based on their historical financial transactions. Using a cleaned dataset of filtered expense transactions (excluding income and transfers), the model aggregates data at the monthly level and applies lag and rolling mean features to capture temporal patterns and seasonality. A Ridge regression model is used as the primary estimator due to its robustness against small datasets and multicollinearity, while a TimeSeriesSplit cross-validation ensures realistic backtesting over time without data leakage. The model outputs a single forecast value representing the expected next-month expenditure. To quantify prediction uncertainty, Quantile Regression is used to generate P50 (median) and P90 (upper-bound) budget bands. These provide an interpretable confidence range that can guide users toward more informed financial planning. The model’s average backtest Mean Absolute Error (MAE) reflects how closely predictions align with historical data, while the budget bands indicate possible spending variability. 
The data collection process involves aggregating user financial transactions from multiple sources, including Plaid API–synced accounts, manually entered records, and seeded synthetic data. All transaction data is stored in a centralized PostgreSQL database to ensure consistency and accessibility. For model training and analysis, the PostgreSQL table data is exported into a CSV file containing the fields: id, name, merchant_name, amount, transactionDate, category_id, category_text, cat_name, and is_income. This exported dataset is then integrated into the expense forecasting model pipeline.

- **Goal**: Forecast next month’s **total expenses**.
- **Run (Exploratory Data Analysis (EDA) notebook)**:
  - `notebooks/expenses.ipynb` → produces forecast table and plots.
- **Run (CLI)**:
  ```bash
  python models/expense_forecast.py
  ```
- **Expected results (baseline)**:
  - ![img.png](img.png)

### B. Monthly Income Forecast
The Income Prediction Model is designed to forecast the upcoming month’s total income for a given user based on historical transaction data. The model analyzes monthly income trends, identifies seasonal or recurring patterns, and estimates the expected inflow for the next period. It uses a time series–based Linear Regression model trained on aggregated income data, leveraging past months’ earnings as lagged predictors.
We collect transactions from multiple integrated data sources-including direct personal investigation, controlled experimentation (seeded/synthetic data), user-entered records, and secondary sources such as Plaid Sandbox/API. All transactions (income and expenses) are consolidated in PostgreSQL in the finance_transactions table. For the income prediction model, we filter to confirmed income rows using isIncome = TRUE. This yields only positive inflows (e.g., wages, deposits, refunds) for a given user. Data is pulled with psycopg2 and pandas.read_sql, then handed to the model pipeline.
```angular2html
PostgreSQL:
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

```
- **Goal**: Forecast next month’s **income** and provide **quantile bands** (P50/P90).
- **Model**: Linear Regression + residual bootstrap for predictive intervals.
- **Run (Exploratory Data Analysis (EDA) notebook)**:
  - `income_forecast.ipynb` → Exploratory Data Analysis (EDA).
- **Run (CLI)**:
  ```bash
  python models/income_forecast.py 
    ```
- **Expected results **:
  - Backtest MAE ≈ 120; Example forecast: P50≈$4,900, P90≈$5,060 (illustrative).
![img_1.png](img_1.png)
- 
### C. Budget Prediction Model
This model predicts the top 10 monthly expense categories based on historical user transactions using time-series trend analysis. It aggregates past spending by category, learns the monthly spending pattern using a Linear Regression model, and forecasts the expected amount for the upcoming month. The output helps users anticipate future expenses and plan budgets more effectively by identifying where most of their money is likely to be spent.
We collect transactions from multiple integrated data sources—including direct personal investigation, controlled experimentation (seeded/synthetic data), user-entered records, and secondary sources such as the Plaid Sandbox/API. All collected transactions are unified into a single structured dataset to ensure consistency and completeness for analysis and model training. The consolidated dataset is then exported to CSV format, enabling seamless preprocessing, visualization, and machine learning model development.
- **Expected results
![img_3.png](img_3.png)
  - 
### D. Coupon/Offer Recommendation
The Enhanced Coupon Recommendation Model is an intelligent text-similarity and rule-based hybrid system designed to accurately link user shopping items to relevant digital coupons. It extends a traditional TF-IDF + k-Nearest Neighbors (kNN) retrieval framework with linguistic normalization, domain-specific heuristics, and adaptive confidence boosting to handle real-world coupon data with noisy or inconsistent descriptions. At its core, the model transforms both coupon text and item names into lemmatized TF-IDF vectors, enabling plural and verb variants (e.g., “cookies” vs. “cookie”) to align naturally. Matching uses cosine similarity between these vectors, reinforced by an 80 % name-match gate that filters out weak candidates based on sanitized displayDescription, shortDescription, and longDescription fields-stripped of prices, quantities, and numeric noise. A lightweight lemmatizer, brand-token recognition, and category guardrails prevent false cross-domain matches (e.g earbuds vs. bourbon). This enhanced architecture combines unsupervised text-retrieval learning with rule-driven domain intelligence, yielding precise, explainable coupon–item matching across groceries, household goods, and branded products while remaining lightweight and easily deployable within retail or personal-finance applications.
The Coupon Recommendation Model integrates live offer data from retailer APIs (https://www.ralphs.com/atlas/v1/savings-coupons/v1/coupons) to automatically identify applicable coupons for user-entered shopping transactions. The data collection workflow ensures that the model always operates on active, real-time coupon information and can provide accurate, store-specific savings recommendations. When a user completes a shopping session - for example, spending $120.00 on Ralphs(www.ralphs.com) - they enter the list of purchased products into the web application. The model then analyzes these product descriptions against the current Ralphs coupon dataset to suggest available offers that can be clipped to save money on future or recurring purchases. This approach is transaction-focused, meaning each shopping event is processed individually rather than as a batch of bulk transactions. It provides a personalized and interactive experience that encourages users to review and utilize digital offers right after shopping.

- **Goal**: Recommend relevant store coupons given a basket or user history.
- **Run (EDA notebook)**: `coupons.ipynb` → top‑N offers and hit‑rate@k.
- **Run (CLI)**:
  ```bash
  python models/coupon_matcher_new.py 
  ```
- **Expected results (baseline)**:
- ![img_2.png](img_2.png)

## 4) Reproducibility & Configuration

- **Config file** (optional): `config.yml`
  ```yaml
  database_url: ${DATABASE_URL}
  models_dir: ml/
  artifacts_dir: artifacts/
  forecast:
    horizon_months: 1
    lags: [1,2,3,6,12]
  category_model:
    min_class_count: 10
    calibrate: true
  ```
- **Determinism**: set seeds for numpy/sklearn; fix train/test split by `GroupShuffleSplit` on `merchant_name` for category model to reduce leakage.
- **Versioning**:
  - Save `.joblib` models + exact `categories.json` used for label mapping.
  - Log `requirements.txt` hash with each artifact.

---

## 5) Results Summary 
The project applies Linear Regression for income and expense forecasting, Gradient Boosting for personalized recommendations, and Random Forest for financial health assessment-each evaluated through time-series cross-validation for accuracy and reliability. The models achieved strong performance, with the expense forecast model showing a Mean Absolute Error of \$646 and income prediction maintaining a deviation of only $121, demonstrating robust predictive capability.

---

## 7) Troubleshooting

- **TF‑IDF/Tokenizer not found**: Make sure you restored the full pipeline via `joblib.load`, not just the estimator.
- **Label mismatch**: Re‑generate `category_text_to_id.json` or ensure the same `categories.json` is used at both train and predict time.
- **Database connectivity**: Verify `DATABASE_URL` and SSL settings. Test with `psql` first.
- **Plaid data gaps**: Re‑run sync; confirm webhook processing and cursor state.
- **Timezone issues**: Normalize all timestamps to UTC before monthly grouping.


## 9) License

This repository is for academic use as part of the CSC 590 Graduate Project. 
