"""FastAPI Application Main Entrypoint for FairLens."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.storage.db import init_db

app = FastAPI(
    title="FairLens API",
    description="Production-Grade AI Model Bias & Algorithmic Fairness Auditor",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite database
init_db()

# Mount API routes
app.include_router(api_router)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "FairLens Auditor Backend"}
