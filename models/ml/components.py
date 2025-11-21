# models/ml/components.py
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextJoiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        def clean(s):
            s = (s or "").lower()
            s = re.sub(r"\s+", " ", s).strip()
            return s
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return (X["name"].fillna("").astype(str) + " " +
                X["merchant_name"].fillna("").astype(str)).apply(clean)
