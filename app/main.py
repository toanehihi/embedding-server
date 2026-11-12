import base64
import logging
import secrets
import struct
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, StrictStr, field_validator

from app.config import Settings
from app.engine import BGEEngine, InputTooLong

logger = logging.getLogger(__name__)
Text = Annotated[StrictStr, Field(min_length=1, max_length=131072)]


class EmbeddingRequest(BaseModel):
    model: StrictStr
    input: Text | Annotated[list[Text], Field(min_length=1, max_length=128)]
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Encoding for dense only; sparse always uses JSON indices/values.",
    )
    output_type: Literal["dense", "sparse", "hybrid"] = Field(
        default="dense",
        description="dense: OpenAI-compatible; sparse/hybrid: BGE-M3 custom response extension.",
    )
    dimensions: Literal[1024] | None = None
    user: str | None = None

    @field_validator("input")
    @classmethod
    def reject_blank(cls, value):
        texts = [value] if isinstance(value, str) else value
        if any(not text.strip() for text in texts):
            raise ValueError("Input strings must not be blank")
        return value


class APIError(Exception):
    def __init__(
        self, status: int, message: str, error_type: str, code: str, headers: dict | None = None
    ):
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code
        self.headers = headers


def create_app(
    settings: Settings | None = None, *, engine_factory: Callable = BGEEngine
) -> FastAPI:
    inference_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        config = settings if settings is not None else Settings()
        app.state.settings = config
        app.state.engine = engine_factory(config)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.engine = None

    app = FastAPI(title="BGE-M3 Embedding Server", version="0.1.0", lifespan=lifespan)
    app.state.ready = False
    bearer = HTTPBearer(auto_error=False)

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.error_type,
                    "param": None,
                    "code": exc.code,
                }
            },
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # Do not echo input texts, keys or Pydantic's raw validation context.
        fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
        return await api_error_handler(
            request,
            APIError(
                400,
                "Invalid request field(s): " + ", ".join(fields),
                "invalid_request_error",
                "invalid_request",
            ),
        )

    def authenticate(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        expected = request.app.state.settings.api_key.get_secret_value()
        if credentials is None or not secrets.compare_digest(
            credentials.credentials.encode(), expected.encode()
        ):
            raise APIError(
                401,
                "Invalid or missing API key",
                "authentication_error",
                "invalid_api_key",
                {"WWW-Authenticate": "Bearer"},
            )

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        if not app.state.ready:
            raise APIError(503, "Model not ready", "server_error", "not_ready")
        return {"status": "ready", "device": app.state.engine.device}

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.settings.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "self-hosted",
                }
            ],
        }

    @app.post("/v1/embeddings", dependencies=[Depends(authenticate)])
    def embeddings(body: EmbeddingRequest):
        config = app.state.settings
        if body.model not in {config.model_name, config.model_id}:
            raise APIError(
                404,
                f"Model not found; use {config.model_name}",
                "invalid_request_error",
                "model_not_found",
            )
        texts = [body.input] if isinstance(body.input, str) else body.input
        if len(texts) > config.max_batch_size:
            raise APIError(
                400,
                f"At most {config.max_batch_size} inputs per request",
                "invalid_request_error",
                "batch_too_large",
            )
        if not inference_lock.acquire(blocking=False):
            raise APIError(
                429,
                "Inference busy; retry with backoff",
                "rate_limit_error",
                "server_busy",
                {"Retry-After": "1"},
            )
        try:
            result = app.state.engine.embed(texts, output_type=body.output_type)
        except InputTooLong as exc:
            raise APIError(400, str(exc), "invalid_request_error", "input_too_long") from exc
        except Exception as exc:
            logger.exception("Embedding inference failed")
            raise APIError(
                503,
                "Inference failed; reduce batch size or check server logs",
                "server_error",
                "inference_failed",
            ) from exc
        finally:
            inference_lock.release()
        data = []
        for index in range(len(texts)):
            item = {"object": "embedding", "index": index}
            if result.vectors is not None:
                vector = result.vectors[index]
                item["embedding"] = (
                    base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
                    if body.encoding_format == "base64"
                    else vector
                )
            if result.sparse_vectors is not None:
                sparse = result.sparse_vectors[index]
                item["sparse_embedding"] = {"indices": sparse.indices, "values": sparse.values}
            data.append(item)
        return {
            "object": "list",
            "model": config.model_name,
            "data": data,
            "usage": {"prompt_tokens": result.prompt_tokens, "total_tokens": result.prompt_tokens},
        }

    return app


app = create_app()
