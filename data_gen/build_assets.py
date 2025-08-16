#!/usr/bin/env python3
import argparse, json, os
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path as _Path

# Load .env from repo root (one directory up from this file's folder)
load_dotenv(_Path(__file__).resolve().parents[1] / ".env", override=False)

def _getenv_float(name, default):
    v = os.getenv(name); 
    return float(v) if v is not None else default

def _getenv_int(name, default):
    v = os.getenv(name); 
    return int(v) if v is not None else default

def _getenv_str(name, default):
    v = os.getenv(name); 
    return v if v is not None else default

def _getenv_date(name, default_iso):
    v = os.getenv(name); 
    from datetime import date as _date
    return _date.fromisoformat(v) if v else _date.fromisoformat(default_iso)

def _getenv_int_list(name, default_list):
    v = os.getenv(name)
    if not v:
        return default_list
    try:
        return [int(x.strip()) for x in v.split(",") if x.strip()]
    except Exception:
        return default_list

from dotenv import load_dotenv

import os
import numpy as np
import pandas as pd

def ensure_cascara_in_topn(
    items_w: pd.DataFrame,
    products_df: pd.DataFrame,
    week: int,
    launch_week: int,
    topn: int = 5,
    max_convert_share: float = 0.08,  # cap: convert at most 8% of cold-drink lines
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Ensure Cascara Cold Brew appears in the weekly top-N by revenue for weeks >= launch.
    We minimally convert some non-Cascara cold-drink lines into Cascara lines.

    Keeps total rows/orders stable; only product_id/unit_price/line_amount on selected lines change.
    """
    if week < launch_week or items_w.empty:
        return items_w

    cascara_name = os.getenv("BB_CASCARA_NAME", "Cascara Cold Brew")
    cascara_cat  = os.getenv("BB_CASCARA_CATEGORY", "Cold Drink")

    # Look up Cascara product id & price
    cascara_row = products_df.loc[products_df["name"] == cascara_name]
    if cascara_row.empty:
        return items_w
    cascara_id = int(cascara_row["product_id"].iloc[0])
    cascara_price = float(cascara_row["unit_price"].iloc[0])

    # Identify cold-drink product ids (including Cascara)
    cold_ids = set(products_df.loc[products_df["category"] == cascara_cat, "product_id"])
    cold_ids.add(cascara_id)

    # Current revenue by product
    rev_by_prod = items_w.groupby("product_id")["line_amount"].sum().sort_values(ascending=False)
    cascara_rev = float(rev_by_prod.get(cascara_id, 0.0))
    if len(rev_by_prod) >= topn:
        threshold = float(rev_by_prod.iloc[topn-1])  # revenue of Nth place
    else:
        threshold = 0.0

    # Already in top-N? done.
    if cascara_rev >= threshold:
        return items_w

    # Revenue gap to close (a tiny epsilon so it clearly beats the Nth)
    gap = max(0.0, threshold - cascara_rev + 0.01)

    items = items_w.copy()

    # Candidate lines to convert: other cold drinks (not Cascara)
    cand_mask = items["product_id"].isin(cold_ids) & (items["product_id"] != cascara_id)
    cand_idx = items.index[cand_mask]
    if len(cand_idx) == 0:
        return items

    # Conversion cap
    max_convert = max(1, int(len(cand_idx) * max_convert_share))

    # Prefer converting lines with larger qty first (fewer edits to reach gap)
    cand = items.loc[cand_idx].copy()
    cand["potential_add"] = cand["qty"] * cascara_price  # what Cascara would contribute
    cand = cand.sort_values(["qty", "potential_add"], ascending=False)

    added = 0.0
    changed = 0
    for idx in cand.index:
        # Convert this line to Cascara
        q = float(items.at[idx, "qty"])
        items.at[idx, "product_id"] = cascara_id
        items.at[idx, "unit_price"] = cascara_price
        items.at[idx, "line_amount"] = q * cascara_price

        added += q * cascara_price
        changed += 1
        if added >= gap or changed >= max_convert:
            break

    return items


def _parse_float_list(s, default):
    try:
        return [float(x.strip()) for x in str(s).split(",")]
    except Exception:
        return default

def boost_cascara_lines(
    items_w: pd.DataFrame,
    products_df: pd.DataFrame,
    week: int,
    launch_week: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Post-process one week's order_items to increase Cascara Cold Brew presence
    in weeks >= launch_week by converting some non-Cascara cold-drink lines.
    """
    if week < launch_week or items_w.empty:
        return items_w

    cascara_name = os.getenv("BB_CASCARA_NAME", "Cascara Cold Brew")
    cascara_cat  = os.getenv("BB_CASCARA_CATEGORY", "Cold Drink")

    # Multipliers relative to week-5 baseline (e.g., [1.00, 1.25, 1.50, 1.75])
    mults = _parse_float_list(os.getenv("BB_CASCARA_WEEKLY_MULTIPLIERS", "1.00,1.25,1.50,1.75"), [1.00,1.25,1.50,1.75])
    idx = max(0, min(week - launch_week, len(mults)-1))
    mult = mults[idx]

    # Look up product ids/prices
    cascara_row = products_df.loc[products_df["name"] == cascara_name]
    if cascara_row.empty:
        return items_w  # nothing to do if cascara isn't in the catalog
    cascara_id = int(cascara_row["product_id"].iloc[0])
    cascara_price = float(cascara_row["unit_price"].iloc[0])

    cold_ids = set(products_df.loc[products_df["category"] == cascara_cat, "product_id"])
    if cascara_id not in cold_ids:
        cold_ids.add(cascara_id)

    items = items_w.copy()

    # Current counts
    is_cascara = items["product_id"] == cascara_id
    curr_cascara = int(is_cascara.sum())

    # If Week 5 is the baseline, target_cascara = curr_cascara * mult
    target_cascara = int(round(curr_cascara * mult))
    to_convert = max(0, target_cascara - curr_cascara)

    if to_convert == 0:
        return items

    # Candidates = other cold drinks (not Cascara)
    candidates_mask = items["product_id"].isin(cold_ids) & (~is_cascara)
    n_cand = int(candidates_mask.sum())
    if n_cand <= 0:
        return items

    n_pick = min(to_convert, n_cand)
    pick_idx = rng.choice(items.index[candidates_mask], size=n_pick, replace=False)

    # Convert picked lines to Cascara: product_id + price + line_amount
    items.loc[pick_idx, "product_id"] = cascara_id
    items.loc[pick_idx, "unit_price"] = cascara_price
    items.loc[pick_idx, "line_amount"] = items.loc[pick_idx, "qty"] * cascara_price

    return items


BRAND = _getenv_str("BB_BRAND", "Borough Brew")
START_DATE = _getenv_date("BB_START_DATE", "2024-03-04")   # Monday
NUM_WEEKS = _getenv_int('BB_NUM_WEEKS', 8)

ORDERS_PER_WEEK_AVG = _getenv_int('BB_ORDERS_PER_WEEK_AVG', 650)
ORDERS_PER_WEEK_JITTER = _getenv_float('BB_ORDERS_PER_WEEK_JITTER', 0.10)
NEW_ORDER_SHARE_TARGET = _getenv_float('BB_NEW_ORDER_SHARE_TARGET', 0.33)

MORNING_HOURS = set(_getenv_int_list('BB_MORNING_HOURS', [7,8,9]))
PROMO_MORNING10_WEEKS = set(_getenv_int_list('BB_PROMO_MORNING10_WEEKS', [3,4]))
PROMO_BUNDLE5_WEEKS   = set(_getenv_int_list('BB_PROMO_BUNDLE5_WEEKS', [6,7]))
LAUNCH_CASCARA_WEEK   = _getenv_int('BB_LAUNCH_CASCARA_WEEK', 5)

BASE_PASTRY_ATTACH = _getenv_float('BB_BASE_PASTRY_ATTACH', 0.25)
BUNDLE5_PASTRY_BOOST = _getenv_float('BB_BUNDLE5_PASTRY_BOOST', 0.12)
FOOD_ATTACH = _getenv_float('BB_FOOD_ATTACH', 0.08)

STORES = [
    ("Waterloo", "Southwark"),
    ("Liverpool Street", "City"),
    ("Oxford Circus", "Westminster"),
    ("King’s Cross", "Camden"),
    ("Canary Wharf", "Tower Hamlets"),
    ("London Bridge", "Southwark"),
    ("Victoria", "Westminster"),
    ("Shoreditch", "Hackney"),
    ("Hammersmith", "Hammersmith & Fulham"),
    ("Camden Town", "Camden"),
]

PRODUCTS = [
    ("Americano", "Coffee", 2.50),
    ("Latte", "Espresso Drink", 3.20),
    ("Cappuccino", "Espresso Drink", 3.10),
    ("Flat White", "Espresso Drink", 3.30),
    ("Espresso", "Coffee", 2.20),
    ("Mocha", "Espresso Drink", 3.50),
    ("Iced Latte", "Cold Drink", 3.40),
    ("Iced Americano", "Cold Drink", 2.70),
    ("Cascara Cold Brew", "Cold Drink", 3.90),  # from Week 5
    ("English Breakfast Tea", "Tea", 2.20),
    ("Earl Grey", "Tea", 2.30),
    ("Green Tea", "Tea", 2.40),
    ("Croissant", "Pastry", 2.10),
    ("Almond Croissant", "Pastry", 2.60),
    ("Pain au Chocolat", "Pastry", 2.50),
    ("Cinnamon Roll", "Pastry", 2.80),
    ("Ham & Cheese Sandwich", "Food", 4.20),
    ("Veggie Sandwich", "Food", 4.00),
    ("Chicken Caesar Wrap", "Food", 4.50),
    ("Porridge", "Food", 3.00),
    ("Yogurt Parfait", "Food", 3.20),
    ("Banana Bread Slice", "Food", 2.30),
]

STORE_WEIGHTS = np.array([1.2, 1.1, 1.2, 1.0, 1.1, 1.1, 1.0, 1.0, 0.8, 0.9]); STORE_WEIGHTS /= STORE_WEIGHTS.sum()

DRINK_WEIGHTS = {"Coffee": 0.25, "Espresso Drink": 0.42, "Cold Drink": 0.18, "Tea": 0.15}

PRODUCT_POP = {
    "Flat White": 1.25, "Latte": 1.20, "Cappuccino": 1.05, "Americano": 1.00, "Espresso": 0.8, "Mocha": 0.9,
    "Iced Latte": 1.0, "Iced Americano": 0.8, "Cascara Cold Brew": 1.05,
    "English Breakfast Tea": 1.0, "Earl Grey": 0.9, "Green Tea": 0.9,
    "Croissant": 1.1, "Almond Croissant": 1.0, "Pain au Chocolat": 1.0, "Cinnamon Roll": 0.9,
    "Ham & Cheese Sandwich": 1.0, "Veggie Sandwich": 0.9, "Chicken Caesar Wrap": 1.0,
    "Porridge": 0.8, "Yogurt Parfait": 0.9, "Banana Bread Slice": 0.95,
}

HOUR_PROFILE = {7:1.00,8:1.30,9:1.15,10:0.80,11:0.75,12:0.85,13:0.90,14:0.70,15:0.75,16:0.80,17:0.75,18:0.60}

@dataclass
class Paths:
    out: Path
    @property
    def products_csv(self): return self.out / "products.csv"
    @property
    def customers_csv(self): return self.out / "customers.csv"
    @property
    def orders_week(self): return self.out / "orders" / "week"
    @property
    def order_items_week(self): return self.out / "order_items" / "week"
    @property
    def stats_root(self): return self.out / "stats"
    @property
    def schema_json(self): return self.out / "schema.json"

def iso(dt: datetime): return dt.strftime("%Y-%m-%dT%H:%M:%S")

def ensure_dirs(p: Paths):
    p.out.mkdir(parents=True, exist_ok=True)
    p.orders_week.mkdir(parents=True, exist_ok=True)
    p.order_items_week.mkdir(parents=True, exist_ok=True)
    (p.stats_root / "orders" / "week").mkdir(parents=True, exist_ok=True)
    (p.stats_root / "orders-by-store" / "week").mkdir(parents=True, exist_ok=True)
    (p.stats_root / "top-products" / "week").mkdir(parents=True, exist_ok=True)
    (p.stats_root / "new-vs-returning" / "week").mkdir(parents=True, exist_ok=True)
    (p.stats_root / "category-mix" / "week").mkdir(parents=True, exist_ok=True)
    (p.out / "sample" / "orders" / "week").mkdir(parents=True, exist_ok=True)
    (p.out / "sample" / "order_items" / "week").mkdir(parents=True, exist_ok=True)

def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True); df.to_csv(path, index=False)

def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

def make_products_df():
    rows = []; pid = 1
    for name, cat, price in PRODUCTS:
        rows.append({"product_id": pid, "name": name, "category": cat, "unit_price": round(price, 2)}); pid += 1
    return pd.DataFrame(rows)

def make_customers_df(rng: np.random.Generator, approx_customers=1200):
    total = approx_customers
    new_customers = int(total * NEW_ORDER_SHARE_TARGET) + rng.integers(-30, 30)
    old_customers = total - new_customers

    start_old = START_DATE - timedelta(days=180)
    old_signups = [start_old + timedelta(days=int(rng.integers(0, 180))) for _ in range(old_customers)]
    new_signups = [START_DATE + timedelta(days=int(rng.integers(0, NUM_WEEKS*7))) for _ in range(new_customers)]
    all_signups = old_signups + new_signups
    rng.shuffle(all_signups) # type:ignore

    rows = [{"customer_id": i+1, "signup_date": d.strftime("%Y-%m-%d"), "country": "UK"} for i, d in enumerate(all_signups)]
    return pd.DataFrame(rows)

def choose_store(rng: np.random.Generator):
    idx = rng.choice(len(STORES), p=STORE_WEIGHTS); return STORES[idx]

def hour_weights_for_week(week: int):
    prof = HOUR_PROFILE.copy()
    if week in PROMO_MORNING10_WEEKS:
        prof[7]*=1.15; prof[8]*=1.25; prof[9]*=1.15
    s = sum(prof.values()); return {h: prof[h]/s for h in prof}

def pick_drink(rng: np.random.Generator, week: int, products_df: pd.DataFrame):
    cats = list(DRINK_WEIGHTS.keys()); wts = np.array([DRINK_WEIGHTS[c] for c in cats]); wts /= wts.sum()
    chosen_cat = rng.choice(cats, p=wts)
    eligible = products_df[products_df["category"].isin(["Coffee","Espresso Drink","Cold Drink","Tea"])]
    if week < LAUNCH_CASCARA_WEEK: eligible = eligible[eligible["name"] != "Cascara Cold Brew"]
    subset = eligible[eligible["category"] == chosen_cat].copy()
    pop = np.array([PRODUCT_POP.get(n,1.0) for n in subset["name"]]); pop /= pop.sum()
    row = subset.sample(n=1, weights=pop, random_state=int(rng.integers(0,1_000_000))).iloc[0]
    return dict(row)

def pick_optional_pastry(rng, has_coffee: bool, in_bundle5: bool, products_df: pd.DataFrame):
    p = BASE_PASTRY_ATTACH
    if in_bundle5 and has_coffee: p = min(1.0, p + BUNDLE5_PASTRY_BOOST)
    if rng.random() < p:
        subset = products_df[products_df["category"] == "Pastry"].copy()
        pop = np.array([PRODUCT_POP.get(n,1.0) for n in subset["name"]]); pop /= pop.sum()
        row = subset.sample(n=1, weights=pop, random_state=int(rng.integers(0,1_000_000))).iloc[0]
        return dict(row)
    return None

def pick_optional_food(rng, products_df: pd.DataFrame):
    if rng.random() < FOOD_ATTACH:
        subset = products_df[products_df["category"] == "Food"].copy()
        pop = np.array([PRODUCT_POP.get(n,1.0) for n in subset["name"]]); pop /= pop.sum()
        row = subset.sample(n=1, weights=pop, random_state=int(rng.integers(0,1_000_000))).iloc[0]
        return dict(row)
    return None

def apply_promotions(week: int, is_morning: bool, items: list):
    out = []; applied_pastry_discount = False
    for it in items:
        price = float(it["unit_price"])
        if week in PROMO_MORNING10_WEEKS and is_morning and it["category"] in ["Coffee","Espresso Drink","Cold Drink","Tea"]:
            price = round(price * 0.90, 2)
        out.append({**it, "unit_price": round(price, 2)})
    if week in PROMO_BUNDLE5_WEEKS:
        any_coffee = any(i["category"] in ["Coffee","Espresso Drink"] for i in out)
        if any_coffee:
            for j in range(len(out)):
                if out[j]["category"] == "Pastry" and not applied_pastry_discount:
                    out[j]["unit_price"] = round(max(0.10, out[j]["unit_price"] - 0.50), 2)
                    applied_pastry_discount = True
                    break
    return out

def gen_orders_for_week(rng, week: int, products_df: pd.DataFrame, customers_df: pd.DataFrame, next_order_id: int):
    base = ORDERS_PER_WEEK_AVG * (1 + rng.uniform(-ORDERS_PER_WEEK_JITTER, ORDERS_PER_WEEK_JITTER))
    n_orders = int(round(base))

    hw = hour_weights_for_week(week); hours = list(hw.keys()); probs = np.array([hw[h] for h in hours]); probs /= probs.sum()

    orders_rows, items_rows = [], []
    win_start = START_DATE; win_end = START_DATE + timedelta(days=NUM_WEEKS*7)
    customers_df = customers_df.copy()
    customers_df["is_new_window"] = customers_df["signup_date"].apply(lambda s: win_start <= datetime.strptime(s,"%Y-%m-%d").date() < win_end)
    new_ids = customers_df[customers_df["is_new_window"]]["customer_id"].to_numpy()
    old_ids = customers_df[~customers_df["is_new_window"]]["customer_id"].to_numpy()
    p_new = NEW_ORDER_SHARE_TARGET

    for _ in range(n_orders):
        day_offset = int(rng.integers(0, 7))
        hour = int(rng.choice(hours, p=probs))
        minute = int(rng.integers(0, 60)); second = int(rng.integers(0,60))
        dt = datetime(START_DATE.year, START_DATE.month, START_DATE.day, hour, minute, second) + timedelta(days=(week-1)*7 + day_offset)
        is_morning = int(hour in MORNING_HOURS)

        store_name, borough = choose_store(rng)

        cid = int(rng.choice(new_ids)) if (rng.random() < p_new and len(new_ids)>0) else int(rng.choice(old_ids))

        # at least one drink
        drink = pick_drink(rng, week, products_df)
        items = [ {"product_id": int(drink["product_id"]), "name": drink["name"], "category": drink["category"], "unit_price": float(drink["unit_price"]), "qty": 1} ]

        pastry = pick_optional_pastry(rng, has_coffee=True, in_bundle5=(week in PROMO_BUNDLE5_WEEKS), products_df=products_df)
        if pastry: items.append({"product_id": int(pastry["product_id"]), "name": pastry["name"], "category": pastry["category"], "unit_price": float(pastry["unit_price"]), "qty": 1}) #type:ignore
        food = pick_optional_food(rng, products_df)
        if food: items.append({"product_id": int(food["product_id"]), "name": food["name"], "category": food["category"], "unit_price": float(food["unit_price"]), "qty": 1}) #type:ignore

        items = apply_promotions(week, bool(is_morning), items)

        for it in items:
            line_amount = round(it["unit_price"] * it["qty"], 2)
            items_rows.append({"order_id": next_order_id, "product_id": it["product_id"], "qty": it["qty"], "unit_price": round(it["unit_price"],2), "line_amount": line_amount, "week": week})
        total_amount = round(sum(ir["line_amount"] for ir in items_rows if ir["order_id"] == next_order_id), 2)

        orders_rows.append({
            "order_id": next_order_id,
            "customer_id": cid,
            "store_name": store_name,
            "borough": borough,
            "order_datetime": iso(dt),
            "week": week,
            "is_morning": is_morning,
            "total_amount": total_amount
        })
        next_order_id += 1

    return pd.DataFrame(orders_rows), pd.DataFrame(items_rows), next_order_id

def emit_schema_json(paths):
    schema = {"brand": BRAND, "tables": {
        "products": ["product_id","name","category","unit_price"],
        "customers": ["customer_id","signup_date","country"],
        "orders": ["order_id","customer_id","store_name","borough","order_datetime","week","is_morning","total_amount"],
        "order_items": ["order_id","product_id","qty","unit_price","line_amount","week"]
    }}
    write_json(schema, paths.schema_json)

def stats_orders_week(ow, iw):
    if len(ow)==0: return {"week": None, "orders": 0, "revenue": 0.0, "aov": 0.0, "morning_share": 0.0, "basket_size": 0.0}
    revenue = float(ow["total_amount"].sum()); aov = float(ow["total_amount"].mean()); morning_share = float(ow["is_morning"].mean())
    basket = float(iw["qty"].sum() / iw["order_id"].nunique()) if len(iw)>0 else 0.0; wk = int(ow["week"].iloc[0])
    return {"week": wk, "orders": int(len(ow)), "revenue": round(revenue,2), "aov": round(aov,2), "morning_share": round(morning_share,4), "basket_size": round(basket,2)}

def stats_orders_by_store(ow):
    if len(ow)==0: return []
    g = ow.groupby(["store_name","borough"])["total_amount"].agg(["count","sum","mean"]).reset_index()
    g = g.rename(columns={"count":"orders","sum":"revenue","mean":"aov"}).sort_values("revenue", ascending=False)
    return [{"store_name": r["store_name"], "borough": r["borough"], "orders": int(r["orders"]), "revenue": round(float(r["revenue"]),2), "aov": round(float(r["aov"]),2)} for _, r in g.iterrows()]

def stats_top_products(iw, products_df, topn=5):
    if len(iw)==0: return []
    g = iw.groupby("product_id")["line_amount"].sum().reset_index()
    merged = g.merge(products_df, on="product_id", how="left").sort_values("line_amount", ascending=False).head(topn)
    return [{"product_id": int(r["product_id"]), "name": r["name"], "category": r["category"], "revenue": round(float(r["line_amount"]),2)} for _, r in merged.iterrows()]

def stats_category_mix(iw, products_df, week: int):
    if len(iw)==0: return []
    merged = iw.merge(products_df[["product_id","category"]], on="product_id", how="left")
    g = merged.groupby("category")["line_amount"].sum().reset_index(); total = g["line_amount"].sum()
    g["share"] = g["line_amount"] / total if total>0 else 0.0; g = g.sort_values("line_amount", ascending=False)
    return [{"week": int(week), "category": r["category"], "revenue": round(float(r["line_amount"]),2), "share": round(float(r["share"]),4)} for _, r in g.iterrows()]

def stats_new_vs_returning(ow, customers_df, week: int):
    if len(ow)==0: return {"week": week, "new_pct": 0.0, "returning_pct": 0.0, "new_count": 0, "returning_count": 0}
    merged = ow.merge(customers_df[["customer_id","signup_date"]], on="customer_id", how="left")
    win_start = START_DATE; win_end = START_DATE + timedelta(days=NUM_WEEKS*7)
    merged["is_new"] = merged["signup_date"].apply(lambda s: win_start <= datetime.strptime(s,"%Y-%m-%d").date() < win_end)
    new_count = int(merged["is_new"].sum()); ret_count = int(len(merged) - new_count); total = max(1, new_count + ret_count)
    return {"week": int(week), "new_pct": round(new_count/total,4), "returning_pct": round(ret_count/total,4), "new_count": new_count, "returning_count": ret_count}

def main():
    ap = argparse.ArgumentParser(description="Generate Borough Brew CSVs and JSON stats.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "assets"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--approx-customers", type=int, default=1200)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    paths = Paths(Path(args.out))
    ensure_dirs(paths)

    products_df = make_products_df(); write_csv(products_df, paths.products_csv)
    customers_df = make_customers_df(rng, approx_customers=args.approx_customers); write_csv(customers_df, paths.customers_csv)

    next_order_id = 1000; all_orders, all_items = [], []
    for w in range(1, NUM_WEEKS+1):
        orders_w, items_w, next_order_id = gen_orders_for_week(
            rng, w, products_df, customers_df, next_order_id
        )

        # Optional first step: smooth growth (if you added this earlier)
        items_w = boost_cascara_lines(items_w, products_df, week=w, launch_week=LAUNCH_CASCARA_WEEK, rng=rng)

        # >>> Ensure Cascara appears in top-5 from launch week onwards
        items_w = ensure_cascara_in_topn(
            items_w=items_w,
            products_df=products_df,
            week=w,
            launch_week=LAUNCH_CASCARA_WEEK,
            topn=5,
            max_convert_share=0.08,
            rng=rng,
        )

        # Recompute order totals after edits
        totals = (
            items_w.groupby("order_id", as_index=False)["line_amount"]
                .sum().rename(columns={"line_amount": "total_amount"})
        ) # type: ignore
        orders_w = (
            orders_w.drop(columns=["total_amount"], errors="ignore")
                    .merge(totals, on="order_id", how="left")
        )
        orders_w["total_amount"] = orders_w["total_amount"].fillna(0.0)

    # Now append/write as before...


        # Append AFTER recomputing totals
        all_orders.append(orders_w); all_items.append(items_w)

        # Write per-week CSVs and tiny samples
        write_csv(orders_w, paths.orders_week / f"{w}.csv")
        write_csv(items_w, paths.order_items_week / f"{w}.csv")
        write_json(orders_w.head(5).to_dict(orient="records"), Path(paths.out) / "sample" / "orders" / "week" / f"{w}.json")
        write_json(items_w.head(5).to_dict(orient="records"), Path(paths.out) / "sample" / "order_items" / "week" / f"{w}.json")

    # Concatenate and write the all-weeks tables
    orders = pd.concat(all_orders, ignore_index=True); items  = pd.concat(all_items, ignore_index=True)
    write_csv(orders, paths.out / "orders" / "all.csv")
    write_csv(items,  paths.out / "order_items" / "all.csv")

    # Stats JSONs per week (now reflect boosted Cascara)
    emit_schema_json(paths)
    for w in range(1, NUM_WEEKS+1):
        ow = orders[orders["week"] == w].copy(); iw = items[items["week"] == w].copy()
        write_json(stats_orders_week(ow, iw), paths.stats_root / "orders" / "week" / f"{w}.json")
        write_json(stats_orders_by_store(ow), paths.stats_root / "orders-by-store" / "week" / f"{w}.json")
        write_json(stats_top_products(iw, products_df, topn=5), paths.stats_root / "top-products" / "week" / f"{w}.json")
        write_json(stats_category_mix(iw, products_df, week=w), paths.stats_root / "category-mix" / "week" / f"{w}.json")
        write_json(stats_new_vs_returning(ow, customers_df, week=w), paths.stats_root / "new-vs-returning" / "week" / f"{w}.json")

    print(f"✅ Generated assets under: {paths.out.resolve()}")


if __name__ == "__main__":
    main()
