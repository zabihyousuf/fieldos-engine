"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

# Create FastAPI app
app = FastAPI(
    title="FieldOS Engine",
    description="5v5 flag football simulation and RL optimization engine",
    version="0.1.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fieldos_engine")

@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    logger.info("FieldOS Engine starting up...")
    logger.info("API documentation available at http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks."""
    logger.info("FieldOS Engine shutting down...")
