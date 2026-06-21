"""FastAPI audit service."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from divergencelens.core.config import DivergenceLensConfig
from divergencelens.sdk.client import AuditResult, DivergenceLens

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DivergenceLens",
    description="Reference-free silent-divergence auditing for LangChain Deep Agents",
    version="0.1.0",
)

_lens: DivergenceLens | None = None


def get_lens() -> DivergenceLens:
    global _lens
    if _lens is None:
        _lens = DivergenceLens(DivergenceLensConfig.from_env())
    return _lens


class AuditRequest(BaseModel):
    run_id: str | None = None
    trace: dict | None = None  # raw LangSmith run JSON


class WebhookPayload(BaseModel):
    run_id: str
    project_name: str | None = None
    event: str = "run_complete"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/audit", response_model=AuditResult)
async def audit_endpoint(req: AuditRequest) -> AuditResult:
    lens = get_lens()

    if req.run_id:
        try:
            return lens.audit_langsmith_run(req.run_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if req.trace:
        try:
            from divergencelens.ingest.trace_normalizer import TraceNormalizer
            normalizer = TraceNormalizer()
            run = normalizer.normalize_from_langsmith(req.trace)
            return lens.audit_run(run)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    raise HTTPException(status_code=400, detail="Provide either run_id or trace")


@app.post("/webhook")
async def webhook_endpoint(payload: WebhookPayload) -> dict:
    """Called by LangSmith webhook on run completion."""
    if payload.event != "run_complete":
        return {"status": "ignored", "event": payload.event}

    try:
        lens = get_lens()
        result = lens.audit_langsmith_run(payload.run_id)
        return {
            "status": "audited",
            "run_id": payload.run_id,
            "divergence_count": len(result.divergences),
            "clean": result.is_clean,
        }
    except Exception as exc:
        logger.error("Webhook audit failed for run %s: %s", payload.run_id, exc)
        return {"status": "error", "detail": str(exc)}


def create_app(config: DivergenceLensConfig | None = None) -> FastAPI:
    global _lens
    _lens = DivergenceLens(config or DivergenceLensConfig.from_env())
    return app
