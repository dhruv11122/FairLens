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

# Serve the prebuilt React dashboard from this same Render web service.
# The /api and /health routes above continue to take precedence.
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def serve_frontend(path: str):
        candidate = (FRONTEND_DIST / path).resolve()
        if candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
