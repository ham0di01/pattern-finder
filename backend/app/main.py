import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.endpoints import analysis
from app.db.session import engine
from app.db.base import Base # Import base to ensure models are registered

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Create tables (For dev only - use Alembic in prod)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Pattern Trader API with Advanced Pattern Recognition",
    version="2.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix=settings.API_V1_STR, tags=["analysis"])

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns API status and version information.
    """
    return {
        "status": "ok",
        "version": "2.0.0",
        "pattern_recognition": "enabled",
        "features": [
            "ensemble_matching",
            "feature_engineering",
            "multiple_similarity_metrics",
            "dtw_support",
            "advanced_configuration"
        ]
    }

@app.get("/")
def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Pattern Trader API v2.0",
        "docs": "/docs",
        "health": "/health",
        "api_v1": settings.API_V1_STR
    }