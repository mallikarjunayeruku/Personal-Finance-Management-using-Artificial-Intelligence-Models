# predict_test.py
import re
import joblib, numpy as np, pandas as pd
from pathlib import Path

# ---- Define the custom pieces exactly as in training ----
def _norm(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def build_description(name=None, merchant=None, extra=None):
    parts = []
    if name: parts.append(_norm(name))
    if merchant: parts.append(_norm(merchant))
    if extra: parts.append(_norm(extra))
    return " | ".join(p for p in parts if p)

# Must have the same class name used during training
class DescriptionBuilder:
    # keep the same signature/attributes as training
    def __init__(self, name_col="name", merchant_col="merchant_name", desc_col="description"):
        self.name_col = name_col
        self.merchant_col = merchant_col
        self.desc_col = desc_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
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

# ---- Load and predict ----
ARTIFACT = "/Users/arrju/Documents/CS595/txn-category-ml/models/ml/category_model.joblib"
bundle = joblib.load(ARTIFACT)
model = bundle["model"]
classes = np.array(bundle["classes"])
text_to_id = bundle["category_text_to_id"]

def predict_any(*, description=None, name=None, merchant=None, top_k=1):
    desc = build_description(name=name, merchant=merchant, extra=description)
    X = pd.DataFrame([{"name": name, "merchant_name": merchant, "description": desc}])
    proba = model.predict_proba(X)[0]
    order = np.argsort(proba)[::-1]
    top_idx = order[:max(1, top_k)]
    top = [(classes[i], float(proba[i])) for i in top_idx]
    best_cat, best_p = top[0]
    best_id = int(text_to_id.get(best_cat, -1))
    return {"top": top, "category_text": best_cat, "category_id": best_id, "confidence": best_p}

def predict(name, merchant):
    out = predict_any(name=name, merchant=merchant, top_k=1)
    return {"category_text": out["category_text"], "category_id": out["category_id"], "confidence": out["confidence"]}

# --- Examples (unchanged calls still work) ---
if __name__ == "__main__":
    print(predict_any(description="Gas",top_k=3))
    print(predict_any(description="Groceries", top_k=3))
    print(predict_any(description="WiFi Bill", top_k=3))
    print(predict_any(description="Dinner Food", top_k=3))
    print(predict_any(description="CSUDH Salary", top_k=3))

    # New: description-only usage
    print(predict_any(description="whole foods market grocery", top_k=3))
