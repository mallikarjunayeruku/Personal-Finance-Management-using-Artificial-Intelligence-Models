import re
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, classification_report

# ===== Paths =====
MODEL_PATH = Path("ml/category_model.joblib")
DATA_PATH = "/Users/arrju/Documents/CS595/txn-category-ml/data/transactions_data.csv"

# ===== Config =====
MIN_PER_CLASS_GLOBAL = 2  # require at least this many samples overall

# ===== Utilities =====
def _norm(x: Optional[str]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s

def build_description(name: Optional[str], merchant_name: Optional[str], extra: Optional[str] = None) -> str:
    """
    Combine whatever we have into ONE description string for modeling.
    We intentionally DO NOT use the true category to avoid leakage.
    """
    parts = []
    if name: parts.append(_norm(name))
    if merchant_name: parts.append(_norm(merchant_name))
    if extra: parts.append(_norm(extra))
    desc = " | ".join([p for p in parts if p])
    return desc if desc else ""  # empty allowed; TF-IDF can handle

class DescriptionBuilder(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer: takes a DataFrame with possible columns
    ['name', 'merchant_name', 'description'] and emits a list/Series
    of combined description strings.
    """
    def __init__(self, name_col="name", merchant_col="merchant_name", desc_col="description"):
        self.name_col = name_col
        self.merchant_col = merchant_col
        self.desc_col = desc_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a pandas DataFrame with some or all of these columns.
        names = X[self.name_col] if self.name_col in X.columns else ""
        merchants = X[self.merchant_col] if self.merchant_col in X.columns else ""
        descs = X[self.desc_col] if self.desc_col in X.columns else ""

        out = []
        for n, m, d in zip(
            names if hasattr(names, "__iter__") else [names]*len(X),
            merchants if hasattr(merchants, "__iter__") else [merchants]*len(X),
            descs if hasattr(descs, "__iter__") else [descs]*len(X),
        ):
            out.append(build_description(n, m, d))
        return pd.Series(out)

# ===== Data Load =====
def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    # keep labeled rows only
    df = df[df["category_text"].notna()].copy()

    # Build a 'description' column for modeling from available text
    # If your CSV already has a free-text description column, we'll fold it in here.
    maybe_desc_col = "description" if "description" in df.columns else None
    df["description"] = [
        build_description(
            row.get("name"),
            row.get("merchant_name"),
            row.get(maybe_desc_col) if maybe_desc_col else None
        )
        for _, row in df.iterrows()
    ]
    return df

# ===== Classifier chooser =====
def _choose_classifier(min_count_train: int):
    base = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga")
    if min_count_train >= 3:
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    elif min_count_train == 2:
        return CalibratedClassifierCV(base, method="sigmoid", cv=2)
    else:
        return base

# ===== Train =====
def train(df: pd.DataFrame):
    print(f"Raw: {len(df)} samples, {df['category_text'].nunique()} categories")

    # 1) Filter ultra-rare classes globally
    vc = df["category_text"].value_counts()
    keep = set(vc[vc >= MIN_PER_CLASS_GLOBAL].index)
    df = df[df["category_text"].isin(keep)].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"After filter: {len(df)} samples, {df['category_text'].nunique()} categories")

    # 2) Grouped split by merchant so we evaluate on unseen merchants
    groups = df["merchant_name"].fillna("NA").astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, df["category_text"], groups))
    tr, va = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

    # Ensure validation only contains classes present in training
    seen = set(tr["category_text"].unique())
    va = va[va["category_text"].isin(seen)].copy()

    # 3) Build text pipeline on DESCRIPTION ONLY (single text channel)
    word_vec = ("word_tfidf", TfidfVectorizer(
        ngram_range=(1,3), sublinear_tf=True, max_df=0.95, strip_accents="unicode"))
    char_vec = ("char_tfidf", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True))
    feats = FeatureUnion([word_vec, char_vec])

    # 4) Choose classifier based on smallest class size in training split
    min_count_train = int(tr["category_text"].value_counts().min())
    clf = _choose_classifier(min_count_train)

    pipe = Pipeline([
        ("desc", DescriptionBuilder()),   # -> emits the single description string
        ("feats", feats),
        ("clf", clf),
    ])

    Xtr, ytr = tr[["name", "merchant_name", "description"]], tr["category_text"]
    Xva, yva = va[["name", "merchant_name", "description"]], va["category_text"]

    pipe.fit(Xtr, ytr)

    # 5) Evaluate
    y_prob = pipe.predict_proba(Xva)
    classes = pipe.named_steps["clf"].classes_
    y_pred = classes[np.argmax(y_prob, axis=1)]

    print(f"Min class count (train): {min_count_train}")
    print("Accuracy:", accuracy_score(yva, y_pred))
    print("Macro F1:", f1_score(yva, y_pred, average="macro"))
    print("Top-3:", top_k_accuracy_score(yva, y_prob, k=3, labels=classes))
    print("\nClassification report:\n", classification_report(yva, y_pred))

    return pipe, classes, tr, va

# ===== Convenience: predict from free text only (optionally include name/merchant) =====
def predict_category(model_bundle: Dict[str, Any],
                     description: Optional[str] = None,
                     name: Optional[str] = None,
                     merchant_name: Optional[str] = None,
                     top_k: int = 3):
    """
    Build the single description from whatever you have and return top-k predictions.
    You can pass ONLY `description`, or just `name`, or just `merchant_name`, or any combo.
    """
    model = model_bundle["model"]
    classes = np.array(model_bundle["classes"])

    # unify into one "description" text
    desc = build_description(name=name, merchant_name=merchant_name, extra=description)
    X = pd.DataFrame([{"name": name, "merchant_name": merchant_name, "description": desc}])

    # predict
    probs = model.predict_proba(X)[0]
    order = np.argsort(probs)[::-1]
    top_idx = order[:max(1, top_k)]
    return [(classes[i], float(probs[i])) for i in top_idx]

# ===== Main =====
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    model, classes, tr, va = train(df)

    # Build mapping category_text -> category_id (dedup text → single id)
    category_text_to_id = (df[["category_text", "category_id"]]
                           .drop_duplicates("category_text")
                           .set_index("category_text")["category_id"]
                           .astype(int)
                           .to_dict())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "classes": list(classes),
        "category_text_to_id": category_text_to_id
    }, MODEL_PATH)

    print(f"✅ Model saved to {MODEL_PATH.resolve()}")

    # Quick smoke test (predict using ONLY description)
    demo = "whole foods market grocery"
    loaded = joblib.load(MODEL_PATH)
    print("Top-3 for:", demo, "->", predict_category(loaded, description=demo, top_k=3))
