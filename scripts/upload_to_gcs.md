# Upload assets to Google Cloud Storage (private bucket)

1) Create a bucket (London region example):

```bash
BUCKET=borough-brew-assets-$(openssl rand -hex 4)
gsutil mb -l europe-west2 gs://$BUCKET
```

2) Sync assets from local:

```bash
# from repo root
gsutil -m rsync -r assets gs://$BUCKET
```

3) Set cache headers (recommended):

```bash
gsutil -m setmeta -h "Cache-Control:public, max-age=86400" gs://$BUCKET/**
```

4) Keep the bucket private (default). Grant your Cloud Run service account `storage.objects.get` on the bucket.

5) Deploy FastAPI to Cloud Run with env vars:

```bash
gcloud run deploy borough-brew-api   --source api   --region europe-west2   --allow-unauthenticated   --set-env-vars STORAGE_MODE=gcs,BUCKET_NAME=$BUCKET,SIGNED_URL_TTL_MIN=30,API_KEY=demo-key-123   --max-instances=3
```
