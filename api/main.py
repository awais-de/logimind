"""FastAPI application exposing LogiMind's /query endpoint."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from agents.orchestrator import Orchestrator
from api.routes.query import router as query_router

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the shared Orchestrator on startup, close it on shutdown."""
    app.state.orchestrator = Orchestrator()
    logger.info("Orchestrator ready")
    yield
    await app.state.orchestrator.close()


app = FastAPI(title="LogiMind", lifespan=lifespan)
app.include_router(query_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness check used by the Streamlit frontend."""
    return {"status": "ok"}
