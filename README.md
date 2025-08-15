# Borough Brew — Session 2 API & Data (Option A)

**Option A** architecture: **FastAPI docs** + short-lived **signed-URL redirects** to a **private Google Cloud Storage (GCS)** bucket.

This repo contains:
- `data_gen/` — generator for **synthetic CSV data** and **JSON summaries** (clean, small, reproducible).
- `api/` — FastAPI app exposing **nice endpoints**; in **local mode** it serves files from `assets/`, in **cloud mode** it redirects to **signed URLs** for your private bucket.
- `assets/` — output location for generated CSV/JSON files (git-ignored).

## 1) Generate data (local)

```bash
cd data_gen
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate CSVs + JSON stats into ../assets
python build_assets.py --seed 42
```

## 2) Run FastAPI locally (no GCS needed)

```bash
cd ../api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export STORAGE_MODE=local
export ASSETS_DIR=../assets
export API_KEY=demo-key-123   # set any string you want students to use

uvicorn main:app --reload --port 8080
# open http://localhost:8080/docs
```

## 3) Upload assets to a **private** GCS bucket

Create a bucket (London region example):
```bash
BUCKET=borough-brew-assets-$(openssl rand -hex 4)
gsutil mb -l europe-west2 gs://$BUCKET
```

Sync assets and set cache headers:
```bash
gsutil -m rsync -r ../assets gs://$BUCKET
gsutil -m setmeta -h "Cache-Control:public, max-age=86400" gs://$BUCKET/**
```

Keep the bucket **private** (default). Cloud Run will sign per-request URLs.

## 4) Deploy to Cloud Run (source deploy — no Dockerfile)

```bash
cd ../api

gcloud run deploy borough-brew-api   --source .   --region europe-west2   --allow-unauthenticated   --set-env-vars STORAGE_MODE=gcs,BUCKET_NAME=$BUCKET,SIGNED_URL_TTL_MIN=30,API_KEY=demo-key-123   --max-instances=3
```

- Cloud Run’s service account must have **`storage.objects.get`** on the bucket.
- Base URL will be something like `https://borough-brew-api-xxxx.a.run.app` (or your custom domain).

## 5) Use the API

- **Header:** `X-API-Key: demo-key-123`
- **Endpoints:**

_Phase 1 (JSON summaries)_
```
GET /schema.json
GET /stats/orders/week/{n}.json
GET /stats/orders-by-store/week/{n}.json
GET /stats/top-products/week/{n}.json
GET /stats/new-vs-returning/week/{n}.json
GET /stats/category-mix/week/{n}.json
GET /sample/orders/week/{n}.json
GET /sample/order_items/week/{n}.json
```

_Phase 2 (CSV raw)_
```
GET /products.csv
GET /customers.csv
GET /orders/week/{n}.csv
GET /order_items/week/{n}.csv
```

`n` is 1–8.

## 6) Notes
- Brand: **Borough Brew**. New product launch in Week 5: **Cascara Cold Brew**.
- Data are clean and consistent; promos baked into line prices.
- For the individual assignment, reuse this architecture and swap in a different dataset.


---

## Using a `.env` file (optional but convenient)

1) Copy `.env.example` → `.env` at the repo root and edit values.
2) The **API** (`api/main.py`) and **data generator** (`data_gen/build_assets.py`) automatically load this file.
3) For cloud deploys, set the same variables as Cloud Run env vars (the `.env` is local-only).



---

## Run scripts

After creating `.env` (or using defaults), you can start the API with:

**macOS/Linux**
```bash
bash scripts/run_local.sh
# or to run in GCS mode (requires BUCKET_NAME in .env)
bash scripts/run_gcs.sh
```

**Windows (PowerShell)**
```powershell
.\scriptsun_local.ps1
# or
.\scriptsun_gcs.ps1
```

Or use the Make targets:
```bash
make data        # generate CSV/JSON
make api-local   # run the API in local mode
make api-gcs     # run the API in GCS mode
```
