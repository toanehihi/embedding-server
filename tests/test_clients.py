import socket
import threading

import pytest
import uvicorn

from app.config import Settings
from app.engine import EmbeddingResult, SparseEmbedding
from app.main import create_app

KEY = "integration-test-key"


class FixtureEngine:
    """Replace only expensive model inference; serve the real application over TCP."""

    device = "cpu"

    def embed(self, texts, output_type="dense"):
        dense = [[1.0] + [0.0] * 1023 for _ in texts] if output_type != "sparse" else None
        sparse = (
            [SparseEmbedding([123 + i, 456 + i], [0.25, 0.75]) for i in range(len(texts))]
            if output_type != "dense"
            else None
        )
        return EmbeddingResult(dense, len(texts) * 3, sparse)


@pytest.fixture
def api_base():
    started = threading.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            started.set()

    app = create_app(
        Settings(api_key=KEY, _env_file=None), engine_factory=lambda _: FixtureEngine()
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = Server(uvicorn.Config(app, log_level="error", ws="none"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        assert started.wait(10), "HTTP server failed to start"
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


@pytest.mark.integration
def test_litellm_embedding_over_http(api_base, monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    result = litellm.embedding(
        model="openai/bge-m3",
        api_base=api_base,
        api_key=KEY,
        input=["hello", "xin chào"],
        encoding_format="float",
        timeout=10,
    )
    assert len(result.data) == 2
    assert result.data[0]["embedding"] == [1.0] + [0.0] * 1023
    assert result.usage.prompt_tokens == 6


@pytest.mark.integration
@pytest.mark.parametrize("output_type", ["dense", "sparse", "hybrid"])
@pytest.mark.parametrize("encoding_format", [None, "float", "base64"])
def test_litellm_custom_outputs_survive_http(api_base, monkeypatch, output_type, encoding_format):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    options = {} if encoding_format is None else {"encoding_format": encoding_format}
    result = litellm.embedding(
        model="openai/bge-m3",
        api_base=api_base,
        api_key=KEY,
        input=["first", "second"],
        extra_body={"output_type": output_type},
        timeout=10,
        **options,
    )
    assert len(result.data) == 2
    assert result.usage.prompt_tokens == 6
    for i, item in enumerate(result.data):
        assert item["index"] == i
        if output_type != "dense":
            assert item["sparse_embedding"] == {
                "indices": [123 + i, 456 + i],
                "values": [0.25, 0.75],
            }
        else:
            assert "sparse_embedding" not in item
        if output_type == "sparse":
            # OpenAI SDK/LiteLLM adds embedding=None when the wire field is omitted.
            assert item.get("embedding") is None
        elif encoding_format == "base64":
            import base64
            import struct

            assert struct.unpack("<1024f", base64.b64decode(item["embedding"])) == (
                (1.0,) + (0.0,) * 1023
            )
        else:
            assert item["embedding"] == [1.0] + [0.0] * 1023


@pytest.mark.integration
def test_openai_sdk_default_base64_decoding(api_base):
    openai = pytest.importorskip("openai")
    with openai.OpenAI(api_key=KEY, base_url=api_base, max_retries=0) as client:
        result = client.embeddings.create(model="bge-m3", input="hello")
    assert result.data[0].embedding == [1.0] + [0.0] * 1023
    assert result.usage.total_tokens == 3
