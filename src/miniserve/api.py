"""HTTP transport for the single inference worker.

Handlers submit CPU-side request descriptions and consume output events. They
never call model.forward(), mutate a KV cache, or wait synchronously for tokens.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import queue
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from miniserve.backends.hf_backend import HFBackend
from miniserve.config import ModelConfig, ServingConfig
from miniserve.engine import InferenceEngine, RejectedRequest


class GenerateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=65536, strict=True)
    max_new_tokens: int = Field(default=64, ge=1, strict=True)
    temperature: Literal[0] = 0
    use_chat_template: bool = True

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value


def _event_bytes(event: dict) -> bytes:
    return ("data: " + json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n\n").encode("utf-8")


async def _stream_events(engine, state, request: Request) -> AsyncIterator[bytes]:
    """Poll bounded queues without leaving blocking queue.get tasks behind."""
    try:
        while True:
            if await request.is_disconnected():
                return
            try:
                event = state.output_queue.get_nowait()
            except queue.Empty:
                if state.done.is_set():
                    # A terminal snapshot also handles engines stopped at shutdown.
                    result = state.result()
                    yield _event_bytes({"type": "error" if result['status'] == 'error' else "done", **result})
                    return
                await asyncio.sleep(0.005)
                continue
            yield _event_bytes(event)
            if event.get("type") in {"done", "error"}:
                return
    finally:
        if not state.done.is_set():
            engine.cancel(state.request_id)


async def _wait_for_result(engine, state, request: Request) -> dict:
    try:
        while not state.done.is_set():
            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="client_disconnected")
            await asyncio.sleep(0.005)
        return state.result()
    finally:
        if not state.done.is_set():
            engine.cancel(state.request_id)


def create_app(
    engine: InferenceEngine | None = None,
    model_config: ModelConfig | None = None,
    serving_config: ServingConfig | None = None,
) -> FastAPI:
    """Create one process owning one model and one dedicated inference worker."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_engine = engine
        if active_engine is None:
            backend = await asyncio.to_thread(HFBackend.from_config, model_config or ModelConfig())
            active_engine = InferenceEngine(backend, serving_config or ServingConfig())
        app.state.engine = active_engine
        active_engine.start()
        try:
            yield
        finally:
            await asyncio.to_thread(active_engine.close)

    app = FastAPI(title="MiniServe", version="0.2.0", lifespan=lifespan)

    def get_engine() -> InferenceEngine:
        active_engine = getattr(app.state, "engine", None)
        if active_engine is None:
            raise HTTPException(status_code=503, detail="model_not_loaded")
        return active_engine

    def submit(body: GenerateBody):
        active_engine = get_engine()
        try:
            state = active_engine.submit(
                body.prompt,
                max_new_tokens=body.max_new_tokens,
                use_chat_template=body.use_chat_template,
            )
        except RejectedRequest as error:
            raise HTTPException(status_code=error.status_code, detail=error.reason) from error
        return active_engine, state

    @app.get("/health")
    async def health():
        active_engine = getattr(app.state, "engine", None)
        if active_engine is None:
            return JSONResponse({"loaded": False, "worker_alive": False, "ready": False}, status_code=503)
        snapshot = active_engine.snapshot()
        ready = bool(snapshot.get("worker_alive", False) and snapshot.get("accepting", False))
        return JSONResponse({
            "loaded": True,
            "worker_alive": snapshot.get("worker_alive", False),
            "ready": ready,
        }, status_code=200 if ready else 503)

    @app.get("/metrics")
    async def metrics():
        return get_engine().snapshot()

    @app.post("/v1/generate")
    async def generate(body: GenerateBody, request: Request):
        active_engine, state = submit(body)
        result = await _wait_for_result(active_engine, state, request)
        if result.get("finish_reason") == "queue_timeout":
            return JSONResponse(result, status_code=408)
        if result.get("status") in {"failed", "error"}:
            return JSONResponse(result, status_code=503)
        if result.get("status") == "cancelled":
            return JSONResponse(result, status_code=499)
        return result

    @app.post("/v1/generate/stream")
    async def generate_stream(body: GenerateBody, request: Request):
        active_engine, state = submit(body)
        return StreamingResponse(
            _stream_events(active_engine, state, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
