import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from dotenv import load_dotenv

# --- Load .env from repo root (works in local & Cloud Run builds if present) ---
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

# --- Configuration (env) ---
STORAGE_MODE = os.getenv("STORAGE_MODE", "local").lower()  # "local" | "gcs"
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "../assets")).resolve()
BUCKET_NAME = os.getenv("BUCKET_NAME")
SIGNED_URL_TTL_MIN = int(os.getenv("SIGNED_URL_TTL_MIN", "30"))

API_KEY = os.getenv("API_KEY", "")
REQUIRE_KEY = bool(API_KEY)

# Optional GCS signer (only needed in gcs mode)
if STORAGE_MODE == "gcs":
    from gcs_signer import make_signed_url

# --- API key auth visible in Swagger ---
api_key_header = APIKeyHeader(name="X-API-Key", description="API key for session 2", auto_error=False)

def require_key(x_api_key: str | None = Depends(api_key_header)):
    """Require X-API-Key when API_KEY is set; otherwise allow open access (teaching mode)."""
    if not REQUIRE_KEY:
        return True
    if x_api_key == API_KEY:
        return True
    raise HTTPException(status_code=401, detail="Invalid or missing API key")

# --- Swagger tags & app metadata ---
tags_metadata = [
    {"name": "Schema & Samples", "description": "Table schema and tiny sample rows for a quick look at the data."},
    {"name": "KPIs", "description": "Precomputed weekly metrics to explore simple patterns."},
    {"name": "Raw CSV tables", "description": "Downloadable CSV data."},
]

app = FastAPI(
    title="Borough Brew API",
    description=(
        "This the documentation for the API developed by the Data Science Team at Borough Brew.\n\n"
        "The API provides a point of access to the data gathered by Borough Brew and stored in a PostgreSQL database. \n \n"
        "The API is accessible with the API key that the team at Borough Brew has shared with you. \n\n"

    ),
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"docExpansion": "list",
                           "defaultModelsExpandDepth": -1},  # expand tag sections
    docs_url="/docs",
    redoc_url="/redoc",
)



# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)

# --- Helpers ---
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

def local_path(rel: str) -> Path:
    """Resolve a path under ASSETS_DIR safely."""
    p = (ASSETS_DIR / rel).resolve()
    if not str(p).startswith(str(ASSETS_DIR)) or not p.is_file():
        raise HTTPException(404, "Not found")
    return p

def respond_json(rel_path: str):
    """Serve JSON either directly (local) or via 307 redirect to a signed GCS URL (gcs)."""
    if STORAGE_MODE == "local":
        return FileResponse(local_path(rel_path), media_type="application/json")
    try:
        url = make_signed_url(BUCKET_NAME, rel_path, content_type="application/json", ttl_minutes=SIGNED_URL_TTL_MIN)  # type: ignore[name-defined]
    except FileNotFoundError:
        raise HTTPException(404, "Not found")
    return RedirectResponse(url, status_code=307)

def respond_csv(rel_path: str, filename: str):
    """Serve CSV either directly (local) or via 307 redirect to a signed GCS URL (gcs)."""
    if STORAGE_MODE == "local":
        return FileResponse(
            local_path(rel_path),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    try:
        url = make_signed_url(BUCKET_NAME, rel_path, content_type="text/csv", ttl_minutes=SIGNED_URL_TTL_MIN)  # type: ignore[name-defined]
    except FileNotFoundError:
        raise HTTPException(404, "Not found")
    return RedirectResponse(url, status_code=307)

# --- Response models for nicer docs ---
class OrdersWeekStats(BaseModel):
    week: int
    orders: int
    revenue: float
    aov: float
    morning_share: float
    basket_size: float | None = None

class StoreKpi(BaseModel):
    store_name: str
    borough: str
    orders: int
    revenue: float
    aov: float

class TopProduct(BaseModel):
    product_id: int
    name: str
    category: str
    revenue: float

class NewReturning(BaseModel):
    week: int
    new_pct: float
    returning_pct: float
    new_count: int
    returning_count: int

class CategoryMix(BaseModel):
    week: int
    category: str
    revenue: float
    share: float

# --- Routes ---

@app.get(
    "/schema.json",
    tags=["Schema & Samples"],
    summary="Returns the schema for the available tables",
    description="This endpoint returns basic information on the schema for the Borough Brew database.",
    responses={
        200: {"description": "OK"},
        307: {"description": "Redirect to signed URL (cloud mode)"},
        401: {"description": "Missing/invalid API key"},
    },
)
def schema(ok: bool = Depends(require_key)):
    return respond_json("schema.json")

# Phase 1: JSON KPIs

@app.get(
    "/stats/orders/week/{n}.json",
    tags=["KPIs"],
    summary="Week KPIs (orders, revenue, average order value, morning share, basket size)",
    description="Precomputed metrics for a single week. Great for a fast, initial exploration.",
    response_model=OrdersWeekStats,
    responses={
        200: {
            "description": "OK",
            "content": {"application/json": {"example": {"week": 3, "orders": 642, "revenue": 2154.8, "aov": 3.36, "morning_share": 0.41, "basket_size": 1.32}}},
        },
        307: {"description": "Redirect to signed URL (cloud mode)"},
        401: {"description": "Missing/invalid API key"},
        404: {"description": "Week out of range (1–8)"},
    },
)
def stats_orders_week(
    n: int = PathParam(ge=1, le=8, description="Week number (1–8)"),
    ok: bool = Depends(require_key),
):
    return respond_json(f"stats/orders/week/{n}.json")

@app.get(
    "/stats/orders-by-store/week/{n}.json",
    tags=["KPIs"],
    summary="Week KPIs by store (orders, revenue, average order value)",
    description="Leaderboard of stores for the selected week.",
    response_model=list[StoreKpi],
    responses={307: {"description": "Redirect to signed URL (cloud mode)"}, 401: {"description": "Missing/invalid API key"}, 404: {"description": "Week out of range (1–8)"}},
)
def stats_by_store(
    n: int = PathParam(ge=1, le=8, description="Week number (1–8)"),
    ok: bool = Depends(require_key),
):
    return respond_json(f"stats/orders-by-store/week/{n}.json")

@app.get(
    "/stats/top-products/week/{n}.json",
    tags=["KPIs"],
    summary="Top products by revenue (week)",
    description="Top 5 products for the selected week, with revenue.",
    response_model=list[TopProduct],
    responses={307: {"description": "Redirect to signed URL (cloud mode)"}, 401: {"description": "Missing/invalid API key"}, 404: {"description": "Week out of range (1–8)"}},
)
def stats_top_products(
    n: int = PathParam(ge=1, le=8, description="Week number (1–8)"),
    ok: bool = Depends(require_key),
):
    return respond_json(f"stats/top-products/week/{n}.json")

@app.get(
    "/stats/new-vs-returning/week/{n}.json",
    tags=["KPIs"],
    summary="New vs returning customers (share & counts)",
    description="Share of orders from newly signed-up customers vs. returning ones, for the week.",
    response_model=NewReturning,
    responses={307: {"description": "Redirect to signed URL (cloud mode)"}, 401: {"description": "Missing/invalid API key"}, 404: {"description": "Week out of range (1–8)"}},
)
def stats_new_vs_returning(
    n: int = PathParam(ge=1, le=8, description="Week number (1–8)"),
    ok: bool = Depends(require_key),
):
    return respond_json(f"stats/new-vs-returning/week/{n}.json")

@app.get(
    "/stats/category-mix/week/{n}.json",
    tags=["KPIs"],
    summary="Revenue share by category (week)",
    description="Category mix for the selected week (e.g., Espresso Drink, Pastry, Food).",
    response_model=list[CategoryMix],
    responses={307: {"description": "Redirect to signed URL (cloud mode)"}, 401: {"description": "Missing/invalid API key"}, 404: {"description": "Week out of range (1–8)"}},
)
def stats_category_mix(
    n: int = PathParam(ge=1, le=8, description="Week number (1–8)"),
    ok: bool = Depends(require_key),
):
    return respond_json(f"stats/category-mix/week/{n}.json")

# Samples (tiny JSON extracts)

@app.get(
    "/sample/orders/week/{n}.json",
    tags=["Schema & Samples"],
    summary="Returns a short sample of the Orders table for the selected week",
    responses={200: {"description": "OK"}, 307: {"description": "Redirect to signed URL (cloud mode)"}, 401: {"description": "Missing/invalid API key"}, 404: {"description": "Week out of range (1–8)"}},
)
def sample_orders(
    n: int = PathParam(ge=1, le=8, description="Week number (1–8)"),
    ok: bool = Depends(require_key),
):
    return respond_json(f"sample/orders/week/{n}.json")

@app.get(
    "/sample/order_items/week/{n}.json",
    tags=["Schema & Samples"],
    summary="Returns a short sample of the Orders table for the selected week",
    responses={200: {"description": "OK"}, 307: {"description": "Redirect to signed URL (cloud mode)"}, 401: {"description": "Missing/invalid API key"}, 404: {"description": "Week out of range (1–8)"}},
)
def sample_order_items(
    n: int = PathParam(ge=1, le=8, description="Week number (1–8)"),
    ok: bool = Depends(require_key),
):
    return respond_json(f"sample/order_items/week/{n}.json")

# Phase 2: CSV slices

@app.get(
    "/products.csv",
    tags=["Raw CSV tables"],
    summary="`products` table (CSV)",
    description="Returns full product table in csv format.",
    responses={
        200: {"description": "CSV content", "content": {"text/csv": {"example": "product_id,name,category,unit_price\n1,Latte,Espresso Drink,3.20"}}},
        307: {"description": "Redirect to signed URL (cloud mode)"},
        401: {"description": "Missing/invalid API key"},
    },
)
def products(ok: bool = Depends(require_key)):
    return respond_csv("products.csv", "products.csv")

@app.get(
    "/customers.csv",
    tags=["Raw CSV tables"],
    summary="`customers` table (CSV)",
    description="Returns full customer table in csv format.",
    responses={
        200: {"description": "CSV content", "content": {"text/csv": {"example": "customer_id,signup_date,country\n1,2024-06-03,UK"}}},
        307: {"description": "Redirect to signed URL (cloud mode)"},
        401: {"description": "Missing/invalid API key"},
    },
)
def customers(ok: bool = Depends(require_key)):
    return respond_csv("customers.csv", "customers.csv")

@app.get(
    "/orders/week/{n}.csv",
    tags=["Raw CSV tables"],
    summary="`orders` (CSV slice by week)",
    description="Join with `order_items` (on `order_id`) and `products` (on `product_id`).",
    responses={
        200: {"description": "CSV content", "content": {"text/csv": {"example": "order_id,customer_id,store_name,borough,order_datetime,week,is_morning,total_amount\n1000,42,Waterloo,Southwark,2024-06-10T08:12:00,3,1,5.50"}}},
        307: {"description": "Redirect to signed URL (cloud mode)"},
        401: {"description": "Missing/invalid API key"},
        404: {"description": "Week out of range (1–8)"},
    },
)
def orders_week(
    n: int = PathParam(ge=1, le=8, description="Week number (1–8)"),
    ok: bool = Depends(require_key),
):
    return respond_csv(f"orders/week/{n}.csv", f"orders_week_{n}.csv")

@app.get(
    "/order_items/week/{n}.csv",
    tags=["Raw CSV tables"],
    summary="`order_items` (CSV slice by week)",
    description="Line items with `qty`, `unit_price`, and `line_amount`. Join to `products` for names/categories.",
    responses={
        200: {"description": "CSV content", "content": {"text/csv": {"example": "order_id,product_id,qty,unit_price,line_amount,week\n1000,1,1,3.20,3.20,3"}}},
        307: {"description": "Redirect to signed URL (cloud mode)"},
        401: {"description": "Missing/invalid API key"},
        404: {"description": "Week out of range (1–8)"},
    },
)
def order_items_week(
    n: int = PathParam(ge=1, le=8, description="Week number (1–8)"),
    ok: bool = Depends(require_key),
):
    return respond_csv(f"order_items/week/{n}.csv", f"order_items_week_{n}.csv")

@app.get(
    "/orders.csv",
    tags=["Raw CSV tables"],
    summary="`orders` (all weeks, CSV)",
    description="All orders in one file.",
    responses={
        200: {"description": "CSV content"},
        307: {"description": "Redirect to signed URL (cloud mode)"},
        401: {"description": "Missing/invalid API key"},
    },
)
def orders_all(ok: bool = Depends(require_key)):
    return respond_csv("orders/all.csv", "orders_all.csv")

@app.get(
    "/order_items.csv",
    tags=["Raw CSV tables"],
    summary="`order_items` (all weeks, CSV)",
    description="All line items in one file",
    responses={
        200: {"description": "CSV content"},
        307: {"description": "Redirect to signed URL (cloud mode)"},
        401: {"description": "Missing/invalid API key"},
    },
)
def order_items_all(ok: bool = Depends(require_key)):
    return respond_csv("order_items/all.csv", "order_items_all.csv")
