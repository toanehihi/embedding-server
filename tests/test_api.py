import base64
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

KEY = "test-key-long-enough"
HEADERS = {"Authorization": f"Bearer {KEY}"}


class FakeEngine:
    device = "cpu"

    def embed(self, texts, output_type="dense"):
        from app.engine import EmbeddingResult, SparseEmbedding

        dense = [[1.0] + [0.0] * 1023 for _ in texts] if output_type != "sparse" else None
        sparse = (
            [SparseEmbedding([123 + i, 456 + i], [0.25, 0.75]) for i in range(len(texts))]
            if output_type != "dense"
            else None
        )
        return EmbeddingResult(dense, len(texts) * 3, sparse)


@pytest.fixture
def client():
    from app.config import Settings
    from app.main import create_app

    settings = Settings(api_key=KEY, max_batch_size=2, _env_file=None)
    with TestClient(create_app(settings, engine_factory=lambda _: FakeEngine())) as client:
        yield client


def embed(client, **kwargs):
    return client.post(
        "/v1/embeddings", headers=HEADERS, json={"model": "bge-m3", "input": "hello", **kwargs}
    )


def test_embedding_matches_openai_contract(client):
    response = embed(client, input=["hello", "xin chào"])
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == "bge-m3"
    assert body["usage"] == {"prompt_tokens": 6, "total_tokens": 6}
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert body["data"][0] == {"object": "embedding", "index": 0, "embedding": [1.0] + [0.0] * 1023}


def test_base64_is_little_endian_float32(client):
    response = embed(client, encoding_format="base64", dimensions=1024, user="test")
    assert response.status_code == 200
    raw = base64.b64decode(response.json()["data"][0]["embedding"])
    assert len(raw) == 4096
    assert struct.unpack("<1024f", raw) == (1.0,) + (0.0,) * 1023


@pytest.mark.parametrize("output_type", ["dense", "sparse", "hybrid"])
def test_requested_outputs_only_and_batch_alignment(client, output_type):
    response = embed(client, input=["first", "second"], output_type=output_type)
    assert response.status_code == 200
    body = response.json()
    assert body["usage"] == {"prompt_tokens": 6, "total_tokens": 6}
    for i, item in enumerate(body["data"]):
        assert item["index"] == i
        if output_type != "sparse":
            assert item["embedding"] == [1.0] + [0.0] * 1023
        else:
            assert "embedding" not in item
        if output_type != "dense":
            assert item["sparse_embedding"] == {
                "indices": [123 + i, 456 + i],
                "values": [0.25, 0.75],
            }
        else:
            assert "sparse_embedding" not in item


def test_hybrid_base64_encodes_only_dense(client):
    response = embed(client, output_type="hybrid", encoding_format="base64")
    assert response.status_code == 200
    item = response.json()["data"][0]
    assert struct.unpack("<1024f", base64.b64decode(item["embedding"])) == (1.0,) + (0.0,) * 1023
    assert item["sparse_embedding"] == {"indices": [123, 456], "values": [0.25, 0.75]}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_authentication_required(client, headers):
    response = client.post(
        "/v1/embeddings", headers=headers, json={"model": "bge-m3", "input": "x"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "payload",
    [
        {"input": ""},
        {"input": "  "},
        {"input": []},
        {"input": ["ok", ""]},
        {"input": [1, 2]},
        {"input": [[1, 2]]},
        {"input": None},
        {"input": ["a", "b", "c"]},
        {"dimensions": 512},
        {"encoding_format": "bad"},
        {"output_type": "bad"},
        {"input": "x" * 131073},
    ],
)
def test_invalid_request_is_openai_error(client, payload):
    response = embed(client, **payload)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_unknown_model_is_not_silently_accepted(client):
    response = embed(client, model="other")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_models_and_health(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=HEADERS).json()["data"][0]["id"] == "bge-m3"


def test_busy_inference_is_rejected_without_blocking_health():
    from app.config import Settings
    from app.main import create_app

    started, release = threading.Event(), threading.Event()

    class BlockingEngine(FakeEngine):
        def embed(self, texts, output_type="dense"):
            started.set()
            assert release.wait(5)
            return super().embed(texts)

    with (
        TestClient(
            create_app(Settings(api_key=KEY), engine_factory=lambda _: BlockingEngine())
        ) as c,
        ThreadPoolExecutor() as executor,
    ):
        request = executor.submit(embed, c)
        try:
            assert started.wait(5)
            assert c.get("/healthz").status_code == 200
            response = embed(c)
            assert response.status_code == 429
            assert response.headers["retry-after"] == "1"
        finally:
            release.set()
        assert request.result().status_code == 200


def test_engine_failure_does_not_leak_internal_details():
    from app.config import Settings
    from app.main import create_app

    class BrokenEngine(FakeEngine):
        def embed(self, texts, output_type="dense"):
            raise RuntimeError("secret internal path")

    with TestClient(
        create_app(Settings(api_key=KEY), engine_factory=lambda _: BrokenEngine())
    ) as c:
        response = embed(c)
        assert response.status_code == 503
        assert "secret internal path" not in response.text


def test_token_limit_error_is_bad_request():
    from app.config import Settings
    from app.engine import InputTooLong
    from app.main import create_app

    class LimitedEngine(FakeEngine):
        def embed(self, texts, output_type="dense"):
            raise InputTooLong("Too many tokens")

    with TestClient(
        create_app(Settings(api_key=KEY), engine_factory=lambda _: LimitedEngine())
    ) as c:
        assert embed(c).status_code == 400


def test_empty_api_key_is_rejected():
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(api_key="", _env_file=None)
