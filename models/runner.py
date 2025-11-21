from coupon_matcher import CouponMatcher

matcher = CouponMatcher("/path/to/ralphs_coupons_clean.csv", prefer_same_brand=True, brand_fallback_margin=0.10)

items = [
    {"name": "Pete & Gerry's Organic Free Range Large Eggs", "price": 7.99},
    {"name": "Simple Truth Natural Cage Free Large Brown Eggs", "price": 6.99},
    {"name": "Persil Ultra Pacs Everyday Clean Intense Fresh Laundry Detergent", "price": 15.49},
]
out = matcher.apply(items)
print(out[["item_name","actual_price","new_price","coupon_code","brand","coupon_title","offer_type","offer_amount","expires"]])
