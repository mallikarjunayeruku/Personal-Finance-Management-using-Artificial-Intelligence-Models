#!/usr/bin/env python3
# coupon_matcher.py — TF-IDF + kNN + semantics + guardrails + boosted similarity + lemmatization
# FINAL BUILD (with Persil/Laundry/Pacs fixes, subset fast path, and 'st' stopword)

import re
import json
from pathlib import Path
from typing import List, Dict, Any
from difflib import SequenceMatcher

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ====================== Helpers & Normalization ====================== #

def _coerce_float(x):
    try:
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "").strip()
        return float(x)
    except Exception:
        return None


def _find_first(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _parse_expiry(d: dict):
    for k in ["expires", "expiration", "expirationDate", "endDate", "end_date", "valid_to"]:
        if k in d and d[k]:
            return str(d[k])
    return ""


def sanitize_text(desc: str) -> str:
    """
    Remove all amounts, numbers, and percentages (e.g., $3.00, 30%, 2/$5, 3 for $10, standalone numbers).
    """
    s = (desc or "")
    s = re.sub(r"\$\s*\d+(?:[.,]\d+)?", " ", s)                                # $3, $3.00
    s = re.sub(r"\b\d+\s*/\s*\$\s*\d+(?:[.,]\d+)?", " ", s)                    # 2/$5
    s = re.sub(r"\b\d+\s*(?:for|4)\s*\$\s*\d+(?:[.,]\d+)?", " ", s, flags=re.I)# 3 for $10
    s = re.sub(r"\b\d+(?:[.,]\d+)?\s*%|\b\d+(?:[.,]\d+)?\s*percent\b", " ", s, flags=re.I)
    s = re.sub(r"\b\d+(?:[.,]\d+)?\b", " ", s)                                 # standalone numbers
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------- Lightweight lemmatizer (rule-based; no external corpora) --------- #
VOWELS = set("aeiou")

def _lemmatize_token(tok: str) -> str:
    """Simple lemmatizer for common retail terms (plural/verb forms)."""
    w = tok.lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("ies"):        # cookies -> cookie
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and any(w.endswith(suf) for suf in ("ses","xes","zes","ches","shes")):
        return w[:-2]                            # boxes -> box, dishes -> dish
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]                               # eggs -> egg, headphones -> headphone
    if len(w) > 4 and w.endswith("ed"):
        w = w[:-2]                               # baked -> bake
    if len(w) > 5 and w.endswith("ing"):        # running -> run, baking -> bake
        base = w[:-3]
        if len(base) >= 3 and base[-1] not in VOWELS and base[-2] == base[-1]:
            base = base[:-1]
        if len(base) >= 2 and base[-1] not in VOWELS:
            base = base + "e"
        return base
    return w

def _tokenize_lemma(s: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9]{2,}", (s or "").lower())
    return [_lemmatize_token(t) for t in toks]


# ====================== Flatten & Load ====================== #

def _flatten_coupon(raw: dict) -> dict:
    d = dict(raw)
    for key in ("coupon", "offer", "attributes"):
        if isinstance(d.get(key), dict):
            d = {**d, **d[key]}
    return {
        "id": str(_find_first(d, ["id", "couponId", "coupon_id", "offerId", "offer_id", "sku"], "")),
        "title": str(_find_first(d, ["title", "name", "description", "shortDescription"], "")),
        "brand": str(_find_first(d, ["brand", "brandName", "brand_name", "manufacturer"], "")),
        "category": str(_find_first(d, ["category", "categoryName", "department", "dept"], "")),
        "discount": _find_first(d, ["discount", "savings", "amount", "value", "offerAmount", "offer_value", "price", "reward"], ""),
        "expires_raw": _parse_expiry(d),
        "displayDescription": str(_find_first(d, ["displayDescription"], "")),
        "shortDescription": str(_find_first(d, ["shortDescription"], "")),
        "longDescription": str(_find_first(d, ["longDescription", "long_description", "fullDescription"], "")),
        "store": str(_find_first(d, ["store", "retailer", "retailerName"], "Ralphs")),
    }

def load_coupons(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Coupons file not found: {path}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p).fillna("")
        for col in ["id","title","brand","category","discount","expires_raw","displayDescription","shortDescription","longDescription"]:
            if col not in df.columns: df[col] = ""
        return df

    if p.suffix.lower() in (".json",".ndjson"):
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ["coupons","data","offers","items","results","list","records"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]; break
            else:
                lists = [v for v in data.values() if isinstance(v, list)]
                items = lists[0] if lists else [data]
        elif isinstance(data, list):
            items = data
        else:
            items = []
        rows = [_flatten_coupon(x) for x in items]
        return pd.DataFrame(rows).fillna("")

    raise ValueError("Unsupported coupons file format; use CSV or JSON.")


# ====================== Guardrail Lexicons ====================== #

STOPWORDS = {
    "the","a","an","with","and","or","for","to","of","on","off","by","from","at","in","per","each","every",
    "black","white","red","blue","green","xl","l","m","s","pack","oz","ct","size","new","save","coupon","offer",
    "st"  # brand abbrev (Simple Truth)
}
ELECTRONICS_TOKENS = {"earbud","earbuds","headphone","headphones","audio","mic","wired","bluetooth","skullcandy","jib","buds"}
SPIRITS_TOKENS     = {"bourbon","whiskey","whisky","vodka","tequila","rum","gin","liqueur","beer","wine","alcohol","spirit","distillery"}
SPIRITS_BRANDS     = {"larceny","makers","maker’s","maker's","jim","beam","wild","turkey","woodford","bulleit",
                      "knob","creek","buffalo","trace","jack","daniel","jack daniel’s","jack daniels"}


def strong_tokens(s: str) -> set:
    return {t for t in _tokenize_lemma(s) if t not in STOPWORDS}

def coarse_domain(text: str) -> str:
    st = strong_tokens(text)
    if st & { _lemmatize_token(t) for t in ELECTRONICS_TOKENS }: return "electronics"
    if st & { _lemmatize_token(t) for t in SPIRITS_TOKENS }:     return "spirits"
    if "egg" in st:              return "eggs_dairy"
    if "detergent" in st or "laundry" in st: return "detergent"
    return "general"

def parse_title_offer(title: str, discount_field: str):
    t = (title or "").strip()
    d = (discount_field or "").strip()

    pct_t = re.search(r"(\d+(?:\.\d+)?)\s*[%％]", t)
    if pct_t:
        return "save_pct", float(pct_t.group(1)), True

    if t.lower().startswith("save"):
        m = re.search(r"(\d+(?:\.\d{1,2})?)", t)
        if m:
            return "save", float(m.group(1)), False
        val = _coerce_float(d)
        if val is not None:
            return "save", val, False
        return "save", None, False

    m = re.search(r"\$?\s*(\d+(?:\.\d{1,2})?)", t)
    if m:
        return "save", float(m.group(1)), False

    return None, None, False

def prepare_coupons(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("")
    parsed = [parse_title_offer(t, d) for t, d in zip(df["title"], df["discount"].astype(str))]
    df["offer_type"] = [p[0] for p in parsed]
    df["offer_title_amount"] = [p[1] for p in parsed]
    df["is_percent"] = [p[2] for p in parsed]
    df["discount_num"] = df["discount"].apply(_coerce_float)

    def choose_amount(r):
        if r["offer_type"] == "save_pct":
            return r["offer_title_amount"]
        if r["offer_type"] == "save":
            return r["discount_num"] if r["discount_num"] is not None else r["offer_title_amount"]
        return r["offer_title_amount"]

    df["offer_amount"] = df.apply(choose_amount, axis=1)

    # Sanitize desc fields for matching
    df["displayDescription_sanitized"] = df["displayDescription"].apply(sanitize_text)
    df["shortDescription_sanitized"]   = df["shortDescription"].apply(sanitize_text)
    df["longDescription_sanitized"]    = df["longDescription"].apply(sanitize_text)

    # Build match text (sanitized descs + brand/title/category)
    for col in ["displayDescription_sanitized","shortDescription_sanitized","longDescription_sanitized","category","brand","title"]:
        if col not in df.columns: df[col] = ""
    df["match_text"] = (
        df["brand"].astype(str) + " " +
        df["title"].astype(str) + " " +
        df["category"].astype(str) + " " +
        df["displayDescription_sanitized"].astype(str) + " " +
        df["shortDescription_sanitized"].astype(str) + " " +
        df["longDescription_sanitized"].astype(str)
    ).str.strip()

    return df


# ====================== Boosted Similarity ====================== #

# Lemmatized keyword inventory (includes pac/pod)
BOOST_KEYWORDS = {
    "egg","detergent","laundry","cookie","meat","lunch","bacon","sausage","cheese","cheddar","milk","yogurt",
    "butter","tortilla","bread","cereal","snack","chip","coffee","tea","juice","earbud","headphone","pac","pod"
}

def boosted_similarity(raw_sim: float, same_brand: bool, item_text: str, coupon_text: str) -> float:
    """
    Boost similarity with strong signals (on lemmatized tokens):
    - +0.35 for same-brand
    - +0.20 if any BOOST_KEYWORDS are present in both item and coupon texts
    """
    bonus = 0.0
    if same_brand:
        bonus += 0.35

    it = set(_tokenize_lemma(item_text))
    ct = set(_tokenize_lemma(coupon_text))
    if len(it & BOOST_KEYWORDS) and len(ct & BOOST_KEYWORDS):
        bonus += 0.20

    return min(1.0, max(0.0, float(raw_sim)) + bonus)


# ====================== Core Model ====================== #

class CouponMatcher:
    def __init__(self, coupons_file: str,
                 prefer_same_brand=True, brand_fallback_margin=0.10,
                 name_gate=0.80, similarity_threshold=0.80):
        """
        prefer_same_brand: prefer same-brand if within +10% of absolute cheapest
        name_gate: 0.80 → if any coupon's (sanitized descs/title) SequenceMatcher ≥ 0.80, restrict to those
        similarity_threshold: 0.80 → boosted cosine sim must meet this to apply coupon
        """
        raw = load_coupons(coupons_file)
        self.coupons = prepare_coupons(raw)
        self.prefer_same_brand = bool(prefer_same_brand)
        self.brand_fallback_margin = float(brand_fallback_margin)
        self.name_gate = float(name_gate)
        self.similarity_threshold = float(similarity_threshold)
        self._fit()

    def _fit(self):
        # Use our lemma tokenizer directly
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=_tokenize_lemma,
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )
        self._X = self.vectorizer.fit_transform(self.coupons["match_text"])
        self.nn = NearestNeighbors(n_neighbors=12, metric="cosine").fit(self._X)

    @staticmethod
    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

    @staticmethod
    def _final_price(price: float, offer_type: str, amt):
        price = float(price)
        if offer_type == "save_pct" and amt is not None:
            return max(0.0, price * (1 - float(amt) / 100.0)), f"Save {float(amt):.0f}%"
        if offer_type == "save" and amt is not None:
            return max(0.0, price - float(amt)), f"Save ${float(amt):.2f}"
        return price, "No usable amount"

    def _apply_gate(self, item_name: str) -> pd.Series:
        c = self.coupons[["displayDescription_sanitized","shortDescription_sanitized","longDescription_sanitized","title"]].fillna("")
        sims = []
        for _, r in c.iterrows():
            s = max(
                self._ratio(item_name, r["displayDescription_sanitized"]),
                self._ratio(item_name, r["shortDescription_sanitized"]),
                self._ratio(item_name, r["longDescription_sanitized"]),
                self._ratio(item_name, r["title"]),
            )
            sims.append(s)
        return pd.Series(sims, index=self.coupons.index) >= self.name_gate

    def best_coupon_for_item(self, name: str, price: float, k: int = 12) -> Dict[str, Any]:
        # 1) 80% name-match gate
        gate_mask = self._apply_gate(name)
        if gate_mask.any():
            cand = self.coupons[gate_mask].copy()
            q = self.vectorizer.transform([name])
            cand_vec = self.vectorizer.transform(cand["match_text"])
            cand["similarity"] = (q @ cand_vec.T).toarray().ravel()
            cand["match_rule"] = "80pct_gate"
        else:
            # 2) kNN fallback
            q = self.vectorizer.transform([name])
            distances, indices = self.nn.kneighbors(q, n_neighbors=min(k, self._X.shape[0]))
            sims = 1 - distances[0]
            cand = self.coupons.iloc[indices[0]].copy()
            cand["similarity"] = sims
            cand["match_rule"] = "knn_fallback"

            # 2a) ≥2 strong-token overlap against sanitized descs + title
            def overlap_ok(row):
                text = " ".join([
                    row.get("displayDescription_sanitized",""),
                    row.get("shortDescription_sanitized",""),
                    row.get("longDescription_sanitized",""),
                    row.get("title",""),
                ])
                return len(strong_tokens(text) & strong_tokens(name)) >= 2

            cand = cand[cand.apply(overlap_ok, axis=1)]

            # 2b) Domain guardrails
            item_domain = coarse_domain(name)
            def domain_ok(row):
                text = " ".join([
                    row.get("brand",""), row.get("title",""), row.get("category",""),
                    row.get("displayDescription_sanitized",""), row.get("shortDescription_sanitized",""),
                    row.get("longDescription_sanitized","")
                ])
                c_domain = coarse_domain(text)
                if item_domain == "electronics" and c_domain == "spirits": return False
                if item_domain == "spirits" and c_domain == "electronics": return False
                return True
            cand = cand[cand.apply(domain_ok, axis=1)]

            # 2c) Spirits brand sanity
            def brand_sanity(row):
                brand = (row.get("brand","") or "").lower()
                if item_domain == "electronics" and any(b in brand for b in SPIRITS_BRANDS): return False
                return True
            cand = cand[cand.apply(brand_sanity, axis=1)]

            if len(cand) > 0:
                cand["match_rule"] = "knn_fallback_strong"

        # Safety: if no candidates remain
        if len(cand) == 0:
            return {
                "item_name": name, "actual_price": round(float(price),2), "new_price": round(float(price),2),
                "coupon_code": "", "coupon_title": "", "brand": "", "offer_type": "", "offer_amount": None,
                "similarity": 0.0, "match_rule": "no_candidates",
                "expires": "", "explanation": "No suitable coupon candidates."
            }

        # 3) Compute final price and notes
        finals, notes = [], []
        for _, r in cand.iterrows():
            f, note = self._final_price(price, r["offer_type"], r["offer_amount"])
            finals.append(f); notes.append(note)
        cand["final_price"] = finals
        cand["note"] = notes

        # 4) Same-brand preference with smart fallback
        item_l = name.lower()
        cand["same_brand"] = cand["brand"].apply(lambda b: (b or "").strip().lower() in item_l if isinstance(b, str) else False)

        cheapest = cand.sort_values(["final_price", "similarity"], ascending=[True, False]).iloc[0]
        chosen = cheapest
        if self.prefer_same_brand and cand["same_brand"].any():
            same = cand[cand["same_brand"]].sort_values(["final_price", "similarity"], ascending=[True, False])
            best_same = same.iloc[0]
            margin_price = float(cheapest["final_price"]) * (1.0 + self.brand_fallback_margin)
            if float(best_same["final_price"]) <= margin_price:
                chosen = best_same

        # 5) Boosted similarity (expanded coupon text) + heuristics
        try:
            _sim_raw = float(chosen.get("similarity", 0))
        except Exception:
            _sim_raw = 0.0

        # Use full coupon text (brand + title + category + sanitized descs)
        _coupon_text = " ".join([
            chosen.get("brand", "") or "",
            chosen.get("title", "") or "",
            chosen.get("category", "") or "",
            chosen.get("displayDescription_sanitized", "") or "",
            chosen.get("shortDescription_sanitized", "") or "",
            chosen.get("longDescription_sanitized", "") or "",
        ]).strip()

        # Brand token heuristic (brand may be in title instead of brand field)
        item_lemmas   = set(_tokenize_lemma(name))
        brand_lemmas  = set(_tokenize_lemma(chosen.get("brand", "") or ""))
        title_lemmas  = set(_tokenize_lemma(chosen.get("title", "") or ""))
        brand_token_match = False
        if brand_lemmas and (item_lemmas & brand_lemmas):
            brand_token_match = True
        elif not brand_lemmas and (item_lemmas & title_lemmas):
            brand_token_match = True

        # Boost with same-brand OR brand-token match
        _sim = boosted_similarity(_sim_raw, brand_token_match or bool(chosen.get("same_brand", False)), name, _coupon_text)

        # Subset fast path: if all item lemmas are in coupon lemmas, lift similarity
        coupon_lemmas = set(_tokenize_lemma(_coupon_text))
        subset_ok = item_lemmas and item_lemmas.issubset(coupon_lemmas)
        if subset_ok:
            _sim = max(_sim, 0.95)

        # Concept uplift: brand match + detergent/laundry + pac/pod on both sides
        concept_detergent = {"detergent","laundry"}
        concept_pacs      = {"pac","pod"}
        if (brand_token_match or bool(chosen.get("same_brand", False))) \
           and (item_lemmas & concept_detergent) and (coupon_lemmas & concept_detergent) \
           and (item_lemmas & concept_pacs) and (coupon_lemmas & concept_pacs):
            _sim = max(_sim, 0.92)

        # 6) Enforce similarity threshold
        if _sim < self.similarity_threshold:
            return {
                "item_name": name,
                "actual_price": round(float(price), 2),
                "new_price": round(float(price), 2),  # unchanged
                "coupon_code": "",
                "coupon_title": "",
                "brand": "",
                "offer_type": "",
                "offer_amount": None,
                "similarity": round(_sim, 3),
                "match_rule": chosen.get("match_rule", ""),
                "expires": "",
                "explanation": f"Ignored — low similarity ({_sim:.2f})",
            }

        # 7) Otherwise, return chosen coupon
        return {
            "item_name": name,
            "actual_price": round(float(price), 2),
            "new_price": round(float(chosen["final_price"]), 2),
            "coupon_code": str(chosen.get("id", "")),
            "coupon_title": chosen["title"],
            "brand": chosen["brand"],
            "offer_type": chosen["offer_type"],
            "offer_amount": float(chosen["offer_amount"]) if chosen["offer_amount"] not in (None, "") else None,
            "similarity": round(float(_sim), 3),  # boosted similarity
            "match_rule": chosen.get("match_rule", ""),
            "expires": chosen.get("expires_raw", ""),
            "explanation": chosen["note"],
        }

    def apply(self, items: List[Dict[str, Any]]) -> pd.DataFrame:
        rows = [self.best_coupon_for_item(it["name"], it["price"]) for it in items]
        return pd.DataFrame(rows, columns=[
            "item_name","actual_price","new_price","coupon_code","coupon_title",
            "brand","offer_type","offer_amount","similarity","match_rule",
            "expires","explanation"
        ])


# ====================== Predefined Run ====================== #

if __name__ == "__main__":
    COUPONS_FILE = "/Users/arrju/Documents/CS595/txn-category-ml/data/coupons.json"

    ITEMS = [
        {"name": "Large Eggs", "price": 7.99},
        {"name": "Oscar Mayer Lunch Meat", "price": 15.99},
        {"name": "Cage Free Large Brown Eggs", "price": 6.99},
        {"name": "Kroger® 100% Apple Juice", "price": 3.19},
        {"name": "Persil Ultra Pacs Everyday Clean Intense Fresh Laundry Detergent", "price": 15.49},
        {"name": "Bakery Fresh Mini Chocolate Chip Cookies", "price": 6.99},
        {"name": "Pepperidge Farm Swirl Bread or Bagels", "price": 5.99},
        {"name": "Jib with Mic Wired Earbuds - Black", "price": 9.99},
        {"name": "Tito's Handmade Vodka", "price": 23.99},
        {"name": "Plain Nonfat Greek Yogurt Tub", "price": 6.99},
    ]

    matcher = CouponMatcher(
        COUPONS_FILE,
        prefer_same_brand=True,
        brand_fallback_margin=0.10,
        name_gate=0.80,
        similarity_threshold=0.80,   # STRICT cutoff (after boosting)
    )

    df = matcher.apply(ITEMS)
    pd.set_option("display.max_colwidth", 120)
    print("\n================= COUPON MATCH RESULTS =================\n")
    print(df[[
        "item_name","actual_price","new_price","coupon_code","brand","coupon_title",
        "offer_type","offer_amount","similarity","match_rule","expires","explanation"
    ]].to_string(index=False))
    print("\n========================================================\n")
