import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class CouponMatcher:
    def __init__(self, coupons_csv: str, prefer_same_brand: bool = True, brand_fallback_margin: float = 0.10):
        """
        coupons_csv: path to the cleaned coupons csv (columns: id,title,brand,category,discount,expires_raw,...)
        prefer_same_brand: try to keep brand if it's still a good deal
        brand_fallback_margin: how much more expensive a same-brand coupon is allowed to be vs. cheapest (e.g., 0.10 = 10%)
        """
        self.coupons = pd.read_csv(coupons_csv).fillna("")
        self.prefer_same_brand = prefer_same_brand
        self.brand_fallback_margin = float(brand_fallback_margin)
        self._prepare()
        self._fit()

    @staticmethod
    def _coerce_float(x):
        try:
            if isinstance(x, str):
                x = x.replace("$","").replace(",","").strip()
            return float(x)
        except Exception:
            return None

    @staticmethod
    def _parse_title_offer(title: str, discount_field: str):
        """
        Returns (offer_kind, amount, pct_flag)
          - offer_kind: 'price' | 'save' | 'save_pct' | None
          - amount: float; if pct, 0–100; if price/save, dollars
          - pct_flag: True when amount is a %
        """
        t = (title or "").strip()
        d = (discount_field or "").strip()

        # Percentage anywhere in title (e.g., "Save 25%", "25% off")
        pct_t = re.search(r'(\d+(?:\.\d+)?)\s*[%％]', t)
        if pct_t:
            return "save_pct", float(pct_t.group(1)), True

        # "Save $X" or "Save X" (assume dollars)
        if t.lower().startswith("save"):
            m = re.search(r'(\d+(?:\.\d{1,2})?)', t)
            if m:
                return "save", float(m.group(1)), False
            # fallback to discount field
            if any(ch in d for ch in "%％"):
                m2 = re.search(r'(\d+(?:\.\d{1,2})?)', d)
                if m2:
                    return "save_pct", float(m2.group(1)), True
            else:
                try:
                    val = float(d.replace("$","").replace(",",""))
                    return "save", val, False
                except Exception:
                    pass
            return "save", None, False

        # Purchase price present (e.g., "$4.00")
        m = re.search(r'(\d+(?:\.\d{1,2})?)', t)
        if m:
            return "price", float(m.group(1)), False

        # Percentage in discount field
        if any(ch in d for ch in "%％"):
            m2 = re.search(r'(\d+(?:\.\d{1,2})?)', d)
            if m2:
                return "save_pct", float(m2.group(1)), True

        return None, None, False

    def _prepare(self):
        c = self.coupons
        c["title"] = c["title"].astype(str)
        c["brand"] = c["brand"].astype(str)
        c["category"] = c["category"].astype(str) if "category" in c.columns else ""
        c["match_text"] = (c["brand"] + " " + c["title"] + " " + c["category"]).str.strip()

        disc_col = "discount" if "discount" in c.columns else None
        discount_series = c[disc_col].astype(str) if disc_col else pd.Series([""]*len(c))

        parsed = [self._parse_title_offer(t, d) for t, d in zip(c["title"], discount_series)]
        c["offer_type"] = [p[0] for p in parsed]
        c["offer_title_amount"] = [p[1] for p in parsed]
        c["is_percent"] = [p[2] for p in parsed]
        c["discount_num"] = c["discount"].apply(self._coerce_float) if "discount" in c.columns else None

        def choose_amount(row):
            if row["offer_type"] == "price" and row["offer_title_amount"] is not None:
                return row["offer_title_amount"]
            if row["offer_type"] == "save_pct" and row["offer_title_amount"] is not None:
                return row["offer_title_amount"]     # keep the percent (0–100)
            if row["offer_type"] == "save":
                # Prefer numeric 'discount' value when available
                return row["discount_num"] if (row["discount_num"] is not None and not pd.isna(row["discount_num"])) else row["offer_title_amount"]
            return row["offer_title_amount"]

        c["offer_amount"] = c.apply(choose_amount, axis=1)

    def _fit(self):
        c = self.coupons
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
        self._X = self.vectorizer.fit_transform(c["match_text"])
        self.nn = NearestNeighbors(n_neighbors=8, metric="cosine").fit(self._X)

    def _compute_final(self, item_price: float, offer_type: str, amt):
        if offer_type == "price" and amt is not None:
            return float(amt), f"Purchase offer — ${float(amt):.2f}"
        if offer_type == "save_pct" and amt is not None:
            newp = max(0.0, item_price * (1 - float(amt)/100.0))
            return newp, f"Save {float(amt):.0f}%"
        if offer_type == "save" and amt is not None:
            newp = max(0.0, item_price - float(amt))
            return newp, f"Save ${float(amt):.2f}"
        return item_price, "No usable amount"

    def best_coupon_for_item(self, item_name: str, item_price: float, k: int = 8):
        q = self.vectorizer.transform([item_name])
        distances, indices = self.nn.kneighbors(q, n_neighbors=min(k, self._X.shape[0]))
        sims = 1 - distances[0]
        candidates = self.coupons.iloc[indices[0]].copy()
        candidates["similarity"] = sims

        # compute final price per candidate
        finals, notes = [], []
        for _, r in candidates.iterrows():
            f, note = self._compute_final(item_price, r["offer_type"], r["offer_amount"])
            finals.append(f); notes.append(note)
        candidates["final_price"] = finals
        candidates["note"] = notes

        # same-brand preference with smart fallback
        item_l = item_name.lower()
        def is_same_brand(b):
            b = (b or "").strip().lower()
            return bool(b) and (b in item_l)
        candidates["same_brand"] = candidates["brand"].apply(is_same_brand)

        cheapest_overall = candidates.sort_values(["final_price","similarity"], ascending=[True, False]).iloc[0]
        chosen = cheapest_overall
        if self.prefer_same_brand and candidates["same_brand"].any():
            same = candidates[candidates["same_brand"]].sort_values(["final_price","similarity"], ascending=[True, False])
            best_same = same.iloc[0]
            margin_price = float(cheapest_overall["final_price"]) * (1.0 + self.brand_fallback_margin)
            if float(best_same["final_price"]) <= margin_price:
                chosen = best_same

        return {
            "item_name": item_name,
            "actual_price": round(item_price, 2),
            "new_price": round(float(chosen["final_price"]), 2),
            "coupon_code": str(chosen.get("id","")),
            "coupon_title": chosen["title"],
            "brand": chosen["brand"],
            "offer_type": chosen["offer_type"],
            "offer_amount": float(chosen["offer_amount"]) if chosen["offer_amount"] is not None else None,
            "similarity": round(float(chosen["similarity"]), 3),
            "expires": chosen.get("expires_raw",""),
            "explanation": chosen["note"],
        }

    def apply(self, items):
        rows = [self.best_coupon_for_item(it["name"], it["price"]) for it in items]
        return pd.DataFrame(rows, columns=[
            "item_name","actual_price","new_price","coupon_code","coupon_title","brand",
            "offer_type","offer_amount","similarity","expires","explanation"
        ])
