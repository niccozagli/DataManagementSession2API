from pathlib import Path
import os
from dotenv import load_dotenv
import uvicorn

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

host = os.getenv("API_HOST", "0.0.0.0")
port = int(os.getenv("API_PORT", "8080"))

if __name__ == "__main__":
    uvicorn.run("main:app", host=host, port=port, reload=True)
