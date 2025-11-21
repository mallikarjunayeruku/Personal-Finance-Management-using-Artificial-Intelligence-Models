
#!/usr/bin/env python3
"""
txnbot_cli.py — CLI ML chatbot over transaction history (patched)

Key fixes in this version:
- Case-insensitive column selection in coerce_columns.pick()
- Support both 'isIncome' and 'is_income' column names
"""

import os
import re
import sys
import json
import joblib
import math
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Utilities
# -----------------------------

# -----------------------------
# Learned Intent Model (optional, overrides RuleNLU if present)
# -----------------------------

class IntentModel:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        self.v = TfidfVectorizer(ngram_range=(1,2), min_df=1, analyzer="word")
        self.clf = LinearSVC()

    def fit(self, texts, labels):
        X = self.v.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self.v.transform(texts)
        return self.clf.predict(X)

def train_intent_model(csv_path):
    df = pd.read_csv(csv_path)
    if not {"text","intent"}.issubset({c.lower() for c in df.columns.map(str.lower)}):
        raise ValueError("nlu_samples.csv must have columns: text,intent")
    # Robust column access
    def col(name):
        return next(c for c in df.columns if c.lower()==name)
    X = df[col("text")].astype(str).tolist()
    y = df[col("intent")].astype(str).tolist()
    model = IntentModel().fit(X, y)
    return model

def save_intent(model, artifact_dir):
    import joblib, os
    joblib.dump(model, os.path.join(artifact_dir, "intent_model.joblib"))

def load_intent(artifact_dir):
    import joblib, os
    p = os.path.join(artifact_dir, "intent_model.joblib")
    if os.path.exists(p):
        return joblib.load(p)
    return None


CATEGORY_ALIASES = {
    "groceries": ["grocery", "groceries", "supermarket", "supermarkets"],
    "rent": ["rent", "landlord"],
    "restaurants": ["restaurant", "restaurants", "dining", "fast food", "coffee"],
    "gas": ["gas", "fuel"],
    "utilities": ["utilities", "water", "electric", "gas bill", "internet", "cable", "wifi"],
    "subscriptions": ["subscription", "netflix", "hulu", "amazon prime", "prime video", "spotify", "apple music", "youtube", "disney", "hbomax"],
    "travel": ["travel", "airline", "flight", "hotel", "uber", "lyft"],
    "coffee": ["coffee", "starbucks", "peets"],
    "shopping": ["shopping", "retail", "store", "walmart", "target", "costco"],
    "bar": ["bar", "pub", "brewery", "liquor"],
    "pharmacy": ["pharmacy", "cvs", "walgreens", "rite aid"],
    "healthcare": ["healthcare", "doctor", "hospital", "clinic"],
    "insurance": ["insurance", "vehicle insurance", "car insurance", "health insurance"],
    "mobile": ["mobile", "cell", "phone", "verizon", "att", "t-mobile"],
    "wifi": ["wifi", "internet", "spectrum", "xfinity", "cox"]
}

SUBSCRIPTION_KEYWORDS = {
"subscription", "renewal", "netflix", "hulu", "prime", "spotify", "apple", "youtube", "hbomax", "disney", "xbox", "playstation"}

DATE_PAT = re.compile(r"(?:\b(20\d{2})-(\d{2})(?:-(\d{2}))?\b)|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})\b", re.I)

def build_reverse_alias():
    rev = {}
    for canon, aliases in CATEGORY_ALIASES.items():
        for a in aliases:
            rev[a.lower()] = canon
    return rev

REVERSE_ALIAS = build_reverse_alias()

REL_PAT = re.compile(r"\b(last|this|next)\s+(month|week|year)\b", re.I)

def parse_bool(x):
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if x is None: return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}

def coerce_columns(df):
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            key = str(c).lower()
            if key in cols: return cols[key]
        return None

    # Rename commonly used columns if absent
    mapping = {}
    if pick("merchantname") is None and pick("merchant_name"):
        mapping[cols["merchant_name"]] = "merchantName"
    if pick("transactiondate") is None and pick("date"):
        mapping[cols["date"]] = "transactionDate"
    df = df.rename(columns=mapping)

    # Refresh cols after rename
    cols = {c.lower(): c for c in df.columns}

    # Ensure required
    for need in ["amount", "transactiondate"]:
        if pick(need) is None:
            raise ValueError(f"Required column '{need}' missing (case-insensitive). Found: {list(df.columns)}")

    # Build combined text
    name_col = pick("name") or pick("description") or pick("memo")
    merch_col = pick("merchantname") or pick("merchant_name")
    df["name"] = df[name_col] if name_col else ""
    if merch_col:
        df["merchantName"] = df[merch_col]
    else:
        if "merchantName" not in df.columns:
            df["merchantName"] = ""

    # Dates
    df["transactionDate"] = pd.to_datetime(df[pick("transactiondate")], errors="coerce", utc=True)
    df["date"] = df["transactionDate"].dt.date
    # Amount sign convention: positive = expense, negative = income (Plaid-like)
    df["amount"] = pd.to_numeric(df[pick("amount")], errors="coerce")
    # Income flag
    income_col = pick("isincome", "is_income")
    if income_col:
        df["isIncome"] = df[income_col].apply(parse_bool)
    else:
        # Infer: negative amounts are income-like
        df["isIncome"] = df["amount"] < 0

    # Category text
    cat_col = pick("category") or pick("category_id") or pick("category_text") or pick("cat_name") or pick("pfm_category") or pick("pfc_primary")
    if cat_col:
        df["category"] = df[cat_col].astype(str)
    else:
        df["category"] = ""

    # Normalize sign: work with absolute spend for expenses
    df["signed_amount"] = df["amount"]
    df["abs_amount"] = df["amount"].abs()
    return df

class TextJoiner(BaseEstimator, TransformerMixin):
    def __init__(self, fields=("name", "merchantName")):
        self.fields = fields
    def fit(self, X, y=None): return self
    def transform(self, X):
        parts = []
        for f in self.fields:
            v = X[f].fillna("")
            parts.append(v.astype(str))
        return (parts[0] + " " + parts[1]).values

# -----------------------------
# Intent Classifier (tiny, optional)
# -----------------------------

class RuleNLU:
    def __init__(self):
        self.intent_words = {
            "spending_summary": ["spend", "spent", "spending", "total"],
            "category_summary": ["grocer", "restaurant", "rent", "gas", "utilities", "coffee", "category"],
            "forecast_expense": ["forecast", "predict"],
            "subscriptions": ["subscription", "subscriptions", "duplicate", "recurring"],
            "top_categories": ["top", "biggest", "largest", "categories"],
            "recommendation": ["save", "recommend", "cheaper"],
        }

    def detect_intent(self, text):
        t = text.lower()
        for intent, words in self.intent_words.items():
            if any(w in t for w in words):
                return intent
        return "spending_summary"

    def extract_entities(self, text):
        t = text.lower()
        # category guess + exact alias keyword
        cat = None
        keyword = None
        for alias, canon in REVERSE_ALIAS.items():
            if alias in t:
                cat = canon
                keyword = alias
                break

        # time period
        period = None
        m = REL_PAT.search(t)
        if m:
            rel, unit = m.groups()
            today = pd.Timestamp.utcnow().normalize()
            if unit == "month":
                if rel == "last":
                    target = (today - pd.offsets.MonthBegin(1))
                    period = (target - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
                elif rel == "this":
                    period = today.strftime("%Y-%m")
                elif rel == "next":
                    period = (today + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
        else:
            m2 = DATE_PAT.search(t)
            if m2:
                y, mo, d, mon_name, y2 = m2.groups()
                if y and mo:
                    period = f"{y}-{mo}"
                elif mon_name and y2:
                    month_map = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
                                 "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
                    period = f"{y2}-{month_map[mon_name[:3].lower()]}"

        # numeric months back
        months = None
        m3 = re.search(r"(\d+)\s*(?:month|months)", t)
        if m3:
            months = int(m3.group(1))

        return {"category": cat, "period": period, "months": months, "keyword": keyword}

# -----------------------------
# Models: Category classifier
# -----------------------------

from sklearn.linear_model import LogisticRegression

def train_category_model(df):
    df_train = df.copy()
    df_train = df_train[df_train["category"].astype(str).str.len() > 0]
    if len(df_train) < 20:
        print("[WARN] Not enough labeled rows for category model; skipping.")
        return None

    X = df_train[["name","merchantName"]]
    y = df_train["category"].astype(str).values

    pipe = Pipeline([
        ("join", TextJoiner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=1)),
    ])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    if len(set(yte)) > 1:
        yhat = pipe.predict(Xte)
        print("[Category Model]\\n", classification_report(yte, yhat, zero_division=0))
    return pipe

# -----------------------------
# Analytics helpers
# -----------------------------

def month_key(ts):
    return pd.Timestamp(ts).strftime("%Y-%m")

def filter_period(df, period=None, months=None, end_inclusive=None):
    if period:
        y, m = map(int, period.split("-"))
        start = pd.Timestamp(year=y, month=m, day=1, tz="UTC")
        end = start + pd.offsets.MonthEnd(1)
    else:
        if end_inclusive is None:
            end_inclusive = (pd.Timestamp.utcnow().normalize() - pd.offsets.MonthBegin(1)) - pd.offsets.Day(1)
        if months is None:
            months = 1
        start = (end_inclusive + pd.offsets.Day(1)) - pd.offsets.MonthBegin(months)
        end = end_inclusive
    m = (df["transactionDate"] >= start) & (df["transactionDate"] <= end)
    return df.loc[m].copy()

def summarize_spend(df, category=None, keyword=None):
    d = df[~df["isIncome"]].copy()
    toks = []
    if category:
        toks.append(str(category).lower())
    if keyword and keyword.lower() not in toks:
        toks.append(keyword.lower())
    if toks:
        pat = "|".join(map(re.escape, toks))
        d = d[
            d["category"].astype(str).str.lower().str.contains(pat, na=False) |
            d["name"].astype(str).str.lower().str.contains(pat, na=False) |
            d["merchantName"].astype(str).str.lower().str.contains(pat, na=False)
        ]
    total = d["abs_amount"].sum()
    return float(total)

def monthly_series(df):
    d = df.copy()
    d = d.set_index("transactionDate").sort_index()
    out = d[~d["isIncome"]].resample("MS")["abs_amount"].sum().astype(float)
    return out

def forecast_next_month(df, lookback=3):
    s = monthly_series(df)
    if len(s.dropna()) == 0:
        return 0.0
    s = s.iloc[:-1] if s.index.max().to_period("M") == pd.Timestamp.utcnow().to_period("M") else s
    if len(s) == 0:
        return 0.0
    k = min(lookback, len(s))
    return float(s.tail(k).mean())

def detect_subscriptions(df, months=6):
    end = (pd.Timestamp.utcnow().normalize() - pd.offsets.MonthBegin(1)) - pd.offsets.Day(1)
    d = filter_period(df, months=months, end_inclusive=end)
    d = d[~d["isIncome"]].copy()
    d["key"] = d["merchantName"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]+","",regex=True)
    d["is_sub_word"] = d["name"].str.lower().apply(lambda x: any(k in x for k in SUBSCRIPTION_KEYWORDS))
    d["ym"] = d["transactionDate"].dt.to_period("M")
    counts = d.groupby(["key"])["ym"].nunique()
    candidates = counts[counts >= 3].index
    cand_df = d[d["key"].isin(candidates)]
    dom_std = cand_df.groupby("key")["transactionDate"].apply(lambda s: s.dt.day.std() or 0.0)
    kw = cand_df.groupby("key")["is_sub_word"].mean()
    amt_mean = cand_df.groupby("key")["abs_amount"].mean()
    score = (counts.loc[candidates].astype(float).rename("months")
            .to_frame()
            .join(dom_std.rename("dom_std"))
            .join(kw.rename("kw_rate"))
            .join(amt_mean.rename("avg_amount")))

    # Ensure numeric dtypes to avoid DatetimeArray arithmetic errors
    score["months"] = pd.to_numeric(score["months"], errors="coerce").fillna(0.0)
    score["dom_std"] = pd.to_numeric(score["dom_std"], errors="coerce").fillna(10.0)
    score["kw_rate"] = pd.to_numeric(score["kw_rate"], errors="coerce").fillna(0.0)
    score["avg_amount"] = pd.to_numeric(score["avg_amount"], errors="coerce").fillna(0.0)

    score["score"] = score["months"] - (score["dom_std"] / 10.0) + score["kw_rate"]
    score = score.sort_values("score", ascending=False)
    out = score.reset_index().rename(columns={"key":"merchant"})
    return out.head(20)

# -----------------------------
# Artifact I/O
# -----------------------------
def save_artifacts(artifact_dir, cat_model, meta):
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    if cat_model is not None:
        joblib.dump(cat_model, os.path.join(artifact_dir, "category_model.joblib"))
    with open(os.path.join(artifact_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

def load_artifacts(artifact_dir):
    cat_path = os.path.join(artifact_dir, "category_model.joblib")
    cat_model = joblib.load(cat_path) if os.path.exists(cat_path) else None
    meta_path = os.path.join(artifact_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return cat_model, meta

# -----------------------------
# CLI Commands
# -----------------------------


def cmd_train_nlu(args):
    model = train_intent_model(args.csv)
    Path(args.artifact).mkdir(parents=True, exist_ok=True)
    save_intent(model, args.artifact)
    print(f"[OK] Trained NLU. Artifacts at: {args.artifact}")


def cmd_train(args):
    df = pd.read_csv(args.csv)
    df = coerce_columns(df)
    cat_model = train_category_model(df)
    meta = {
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "csv_path": os.path.abspath(args.csv),
    }
    save_artifacts(args.artifact, cat_model, meta)
    print(f"[OK] Trained. Artifacts at: {args.artifact}")

def cmd_spend(args):
    _, meta = load_artifacts(args.artifact)
    df = pd.read_csv(args.csv) if args.csv else pd.DataFrame()
    if not df.empty:
        df = coerce_columns(df)
    else:
        if "csv_path" in meta and os.path.exists(meta["csv_path"]):
            df = coerce_columns(pd.read_csv(meta["csv_path"]))
        else:
            print("Provide --csv or train with a CSV first.")
            sys.exit(2)

    if args.period or args.months:
        d = filter_period(df, period=args.period, months=args.months)
    else:
        d = df

    total = summarize_spend(d, args.category)
    label = args.period or f"last {args.months or 1} month(s)"
    cat = args.category or "all categories"
    print(f"Spend in {label} for {cat}: ${total:,.2f}")

def cmd_topcats(args):
    _, meta = load_artifacts(args.artifact)
    df = pd.read_csv(args.csv) if args.csv else pd.DataFrame()
    if df.empty and "csv_path" in meta and os.path.exists(meta["csv_path"]):
        df = pd.read_csv(meta["csv_path"])
    if df.empty:
        print("Provide --csv or train with a CSV first."); sys.exit(2)
    df = coerce_columns(df)
    d = filter_period(df, months=args.months)
    x = d[~d["isIncome"]].groupby("category")["abs_amount"].sum().sort_values(ascending=False).head(args.top)
    print("Top categories:")
    for k, v in x.items():
        print(f"  - {k}: ${v:,.2f}")

def cmd_forecast(args):
    _, meta = load_artifacts(args.artifact)
    df = pd.read_csv(args.csv) if args.csv else pd.read_csv(meta["csv_path"])
    df = coerce_columns(df)
    pred = forecast_next_month(df, lookback=args.lookback)
    print(f"Next month forecast (avg of last {args.lookback}): ${pred:,.0f}")

def cmd_subscriptions(args):
    _, meta = load_artifacts(args.artifact)
    df = pd.read_csv(args.csv) if args.csv else pd.read_csv(meta["csv_path"])
    df = coerce_columns(df)
    cand = detect_subscriptions(df, months=args.months)
    if cand.empty:
        print("No likely subscriptions found.")
        return
    print("Likely subscriptions:")
    for _, row in cand.iterrows():
        print(f"  - {row['merchant'] or '[unknown]'} | ${row['avg_amount']:.2f}/mo | cadence_months={int(row['months'])} | score={row['score']:.2f}")

def cmd_ask(args):
    cat_model, meta = load_artifacts(args.artifact)
    df = pd.read_csv(args.csv) if args.csv else pd.read_csv(meta["csv_path"])
    df = coerce_columns(df)
    nlu = RuleNLU()
    intent_model = load_intent(args.artifact)
    if intent_model is not None:
        intent = intent_model.predict([args.question])[0]
    else:
        intent = nlu.detect_intent(args.question)
    slots = nlu.extract_entities(args.question)

    if intent in {"spending_summary", "category_summary"}:
        d = filter_period(df, period=slots.get("period"), months=slots.get("months", 1))
        cat = slots.get("category")
        kw = slots.get("keyword")
        total = summarize_spend(d, cat, kw)
        label = slots.get("period") or f"last {slots.get('months',1)} month(s)"
        cat_label = kw or cat or "all categories"
        print(f"You spent ${total:,.2f} in {label} for {cat_label}.")
        if args.debug:
            m = d.copy()
            toks = [t for t in [cat, kw] if t]
            if toks:
                import re as _re
                pat = _re.compile("|".join(map(_re.escape, [t.lower() for t in toks])))
                mm = m[(m['name'].astype(str).str.lower().str.contains(pat)) | (m['merchantName'].astype(str).str.lower().str.contains(pat)) | (m['category'].astype(str).str.lower().str.contains(pat))]
                print("\n[debug] matched rows (up to 10):")
                print(mm[['transactionDate','name','merchantName','category','amount']].head(10).to_string(index=False))
        return

    if intent == "forecast_expense":
        pred = forecast_next_month(df, lookback=3)
        print(f"Expected next-month expense: ${pred:,.0f}.")
        return

    if intent == "subscriptions":
        cand = detect_subscriptions(df, months=slots.get("months", 6))
        if cand.empty:
            print("No likely subscriptions found.")
        else:
            print("Likely subscriptions:")
            for _, row in cand.iterrows():
                print(f"  - {row['merchant'] or '[unknown]'} | ${row['avg_amount']:.2f}/mo | cadence_months={int(row['months'])}")
        return

    if intent == "top_categories":
        d = filter_period(df, months=slots.get("months", 3))
        x = d[~d["isIncome"]].groupby("category")["abs_amount"].sum().sort_values(ascending=False).head(5)
        print("Top categories:")
        for k, v in x.items():
            print(f"  - {k}: ${v:,.2f}")
        return

    print("Sorry, I couldn't understand. Try: 'How much did I spend last month on groceries?'")

def build_parser():
    p = argparse.ArgumentParser(prog="txnbot_cli", description="ML chatbot over transaction history")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train models from CSV")
    p_train.add_argument("--csv", required=True, help="Path to transactions CSV")
    p_train.add_argument("--artifact", default="./artifacts", help="Output directory for models")
    p_train.set_defaults(func=cmd_train)

    p_spend = sub.add_parser("spend", help="Total spend for a period (optionally category)")
    p_spend.add_argument("--artifact", required=True)
    p_spend.add_argument("--csv", help="Optional CSV (if omitted, uses meta.csv_path)")
    p_spend.add_argument("--period", help="YYYY-MM (e.g., 2025-09)")
    p_spend.add_argument("--months", type=int, help="Last N months (default 1)")
    p_spend.add_argument("--category", help="Category keyword (e.g., groceries)")
    p_spend.set_defaults(func=cmd_spend)

    p_top = sub.add_parser("topcats", help="Top categories over last N months")
    p_top.add_argument("--artifact", required=True)
    p_top.add_argument("--csv")
    p_top.add_argument("--months", type=int, default=3)
    p_top.add_argument("--top", type=int, default=5)
    p_top.set_defaults(func=cmd_topcats)

    p_fc = sub.add_parser("forecast", help="Naive next-month expense forecast")
    p_fc.add_argument("--artifact", required=True)
    p_fc.add_argument("--csv")
    p_fc.add_argument("--lookback", type=int, default=3)
    p_fc.set_defaults(func=cmd_forecast)

    p_subs = sub.add_parser("subscriptions", help="Detect likely subscriptions")
    p_subs.add_argument("--artifact", required=True)
    p_subs.add_argument("--csv")
    p_subs.add_argument("--months", type=int, default=6)
    p_subs.set_defaults(func=cmd_subscriptions)

    p_ask = sub.add_parser("ask", help="Free-text question (tiny NLU)")
    p_ask.add_argument("--artifact", required=True)
    p_ask.add_argument("--csv")
    p_ask.add_argument("--debug", action="store_true")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)


    p_nlu = sub.add_parser("train-nlu", help="Train supervised intent model from CSV (text,intent)")
    p_nlu.add_argument("--csv", required=True, help="Path to nlu_samples.csv")
    p_nlu.add_argument("--artifact", required=True)
    p_nlu.set_defaults(func=cmd_train_nlu)

    return p

def main(argv=None):
    argv = argv or sys.argv[1:]
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
