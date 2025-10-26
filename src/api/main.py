"""FastAPI main application for AGV2.

Minimal working example with dashboard metrics, positions, and inference.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import inference
from api.inference import InferenceEngine
from api.models import HealthResponse
from api.routers import dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting AGV2 API server...")

    # Initialize global inference engine
    inference.inference_engine = InferenceEngine(
        policy_path="models/policies/ppo_trading_env.zip",
        encoder_path="models/encoders/encoder_best.pt",
        device="cpu",
    )

    # Load models
    try:
        await inference.inference_engine.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load models: {e}. API will run with limited functionality.")

    yield

    # Shutdown
    logger.info("Shutting down AGV2 API server...")
    await inference.inference_engine.cleanup()


# Create FastAPI app
app = FastAPI(
    title="AGV2 Trading API",
    description="AI-powered trading system with RL-based policy",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server (CRA)
        "http://localhost:3001",  # Alternative port
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Vite alternative port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        API health status and model loading state
    """
    engine = inference.inference_engine

    return HealthResponse(
        status="healthy",
        api_version="1.0.0",
        model_loaded=engine.is_ready() if engine else False,
        encoder_loaded=engine.is_encoder_ready() if engine else False,
        timestamp=datetime.now(),
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AGV2 Trading API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
