# BGE-M3 Embedding Server

A self-hosted embedding service for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), built with FastAPI and FlagEmbedding. Provides dense, sparse, and hybrid representations through a single HTTP API, with CPU and NVIDIA GPU support.

Dense embeddings use an OpenAI-compatible interface. Sparse and hybrid outputs extend that interface with a custom response field and are accessible through LiteLLM.

## Features

- **Dense embeddings:** 1024-dimensional, L2-normalized vectors.
- **Sparse embeddings:** learned BGE-M3 lexical weights indexed by vocabulary token ID.
- **Hybrid output:** dense and sparse representations from the same encode call.
- **Multilingual input:** individual strings or batches, with stable output ordering.
- **CPU/GPU execution:** automatic device selection, FP32 on CPU, optional FP16 on CUDA.
- **Controlled inference:** per-request batch and token limits, explicit overload responses, and no silent input truncation.
- **Container distribution:** one Linux AMD64 image for both CPU and GPU deployments, published to GitHub Container Registry.

## API

| Method | Endpoint | Authentication | Description |
| --- | --- | --- | --- |
| `POST` | `/v1/embeddings` | Bearer API key | Generate the requested embedding representations |
| `GET` | `/v1/models` | Bearer API key | List the configured public model name |
| `GET` | `/healthz` | None | Process liveness |
| `GET` | `/readyz` | None | Model readiness and selected device |
| `GET` | `/docs` | None | Interactive OpenAPI documentation |

### Embedding request

```json
{
  "model": "bge-m3",
  "input": ["A white cotton T-shirt", "A lightweight summer shirt"],
  "output_type": "hybrid",
  "encoding_format": "float"
}
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | String | Required | Configured model name or model ID |
| `input` | String or array of strings | Required | Non-blank text inputs |
| `output_type` | `dense`, `sparse`, `hybrid` | `dense` | Representations to compute and return |
| `encoding_format` | `float`, `base64` | `float` | Dense vector encoding; sparse output remains JSON |
| `dimensions` | Integer | Omitted | Only `1024` is accepted; dimensionality reduction is not supported |
| `user` | String | Omitted | Accepted for client compatibility; not used by inference |

Each input is limited to 131072 characters before tokenization. Batch size, sequence length, and aggregate token limits are configurable.

### Embedding response

Responses contain `object: "list"`, `model`, `data`, and `usage`. Each item in `data` includes `object: "embedding"` and the original input `index`.

| Output type | Item fields | Representation |
| --- | --- | --- |
| `dense` | `embedding` | Array of 1024 floats, or a base64 string |
| `sparse` | `sparse_embedding` | Object containing parallel `indices` and `values` arrays |
| `hybrid` | `embedding`, `sparse_embedding` | Both representations |

Sparse indices are ascending integer token IDs from the BGE-M3 vocabulary. Values are positive learned weights, not BM25 scores or L2-normalized values. Repeated token IDs use their maximum weight; special tokens are excluded. Empty sparse vectors contain two empty arrays.

Base64 dense vectors use little-endian float32 encoding. Sparse-only responses omit `embedding` and do not compute dense output. Token usage includes special tokens and is counted once per input, including in hybrid mode.

### Error behavior

Errors use an OpenAI-style `error` object with `message`, `type`, `param`, and `code` fields.

| Status | Meaning |
| --- | --- |
| `400` | Invalid request or exceeded batch/token limits |
| `401` | Missing or invalid API key |
| `404` | Unknown model |
| `429` | Inference already in progress; includes `Retry-After: 1` |
| `503` | Inference failed or model not ready |

## LiteLLM compatibility

The LiteLLM provider is `openai/`; the API base ends in `/v1`, not `/v1/embeddings`.

```python
import os

from litellm import embedding

response = embedding(
    model="openai/bge-m3",
    api_base=os.environ["BGE_API_BASE"],
    api_key=os.environ["BGE_API_KEY"],
    input=["A white cotton T-shirt"],
    encoding_format="float",
    extra_body={"output_type": "hybrid"},
    timeout=120,
)

dense = response.data[0]["embedding"]
sparse = response.data[0]["sparse_embedding"]
```

Dense output is compatible with the standard embeddings contract. Sparse and hybrid output are **custom extensions**, not part of the OpenAI embeddings specification. LiteLLM SDK integration tests cover all output types with default, float, and base64 encoding. For sparse-only output, the SDK may add `embedding=None` while preserving `sparse_embedding`.

Proxy middleware or downstream clients that enforce a strict OpenAI response schema may discard custom fields. The included SDK tests do not validate every LiteLLM Proxy configuration.

Reference integrations: [Python client](examples/litellm_client.py) and [LiteLLM Proxy configuration](examples/litellm.yaml).

## Runtime architecture

```text
Client / LiteLLM
       |
       v
FastAPI: authentication, validation, admission control
       |
       v
FlagEmbedding: BGE-M3 encoder and learned representation heads
       |
       +-- dense
       +-- sparse
       +-- dense + sparse
```

The model loads once during application startup. The service accepts requests only after initialization completes; initialization failure prevents startup.

Each process admits one inference request at a time. Inputs within that request are processed in configurable microbatches. Concurrent inference requests receive `429`; there is no request queue or cross-request dynamic batching. Health endpoints remain asynchronous.

A single worker avoids duplicate model allocations. Additional workers or replicas require separate RAM/VRAM capacity. CPU and GPU results may differ slightly because of numerical precision.

Remote model assets are resolved into a common Hugging Face snapshot. Both `sparse_linear.pt` and `colbert_linear.pt` must be present alongside the encoder and tokenizer; incomplete auxiliary heads cause startup failure rather than random sparse-head initialization. `EMBEDDING_MODEL_REVISION` can pin remote assets to a commit.

### CPU and GPU behavior

The published image contains PyTorch CUDA 12.8 and supports CPU fallback with `EMBEDDING_DEVICE=auto`. GPU containers additionally require a compatible host NVIDIA driver, NVIDIA Container Toolkit, and GPU device exposure such as `--gpus all`.

| Device setting | Behavior |
| --- | --- |
| `auto` | Uses CUDA when available to PyTorch; otherwise uses CPU |
| `cpu` | Uses CPU even when a GPU is available |
| `cuda` | Requires CUDA; unavailable CUDA causes startup failure |

Device auto-selection happens inside the application. Docker GPU reservations are separate: requesting GPUs on a host without a configured NVIDIA runtime may prevent container creation before the application starts.

### Container configuration

| Artifact | Purpose |
| --- | --- |
| [Dockerfile](Dockerfile) | Shared CPU/GPU image by default; `BGE_EXTRA=cpu` selects a smaller CPU-only build |
| [compose.yaml](compose.yaml) | Source-based CPU build and local runtime configuration |
| [compose.gpu.yaml](compose.gpu.yaml) | Source-build override selecting CUDA dependencies, exposing GPUs, and forcing `EMBEDDING_DEVICE=cuda` |
| [compose.deploy.yaml](compose.deploy.yaml) | Registry deployment with embedding service and Caddy HTTPS proxy |
| [Caddyfile](Caddyfile) | HTTPS reverse proxy configuration, mounted read-only into Caddy |
| [compose.deploy.gpu.yaml](compose.deploy.gpu.yaml) | Registry deployment override exposing NVIDIA GPUs; device selection remains controlled by `.env` |
| [.env.example](.env.example) | Runtime environment reference |

Source-build and registry-deployment Compose files are separate alternatives. Each GPU override is combined with its corresponding base file. GPU configurations require Docker Compose 2.30 or later and a configured NVIDIA runtime.

Registry deployment command references:

```bash
# CPU deployment or update
docker compose -f compose.deploy.yaml up -d

# GPU deployment or update
docker compose -f compose.deploy.yaml -f compose.deploy.gpu.yaml up -d
```

The registry deployment reads `.env`, defaults to `ghcr.io/toanehihi/embedding-server:latest`, and pulls the image on each `up`. `EMBEDDING_IMAGE` overrides the image reference with a tag or digest. Private packages require registry authentication. The named volume `bge-m3-model-cache` preserves weights and is compatible with the direct Docker deployment cache; `down -v` removes it.

Only Caddy publishes host ports: TCP 80 and 443. The embedding service is reachable internally at `embedding:8000`, not directly on the EC2 host. Existing services using ports 80/443 must be stopped or reconfigured before deployment.

The embedding container runs as a non-root user. Model weights are downloaded at startup rather than embedded in the image. The image cache location is `/home/app/.cache/huggingface`; a persistent volume preserves downloads across container replacements. Initial startup requires network access unless a complete snapshot is already cached. Offline cache operation is supported through `HF_HUB_OFFLINE=1`.

The image healthcheck targets `/readyz`. Docker marks failed healthchecks as unhealthy but does not restart a container solely because of that state.

## Configuration

Settings are loaded from environment variables or `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_API_KEY` | Required | Bearer secret with at least 16 characters |
| `EMBEDDING_MODEL_ID` | `BAAI/bge-m3` | BGE-M3 repository ID or complete local model directory |
| `EMBEDDING_MODEL_REVISION` | Unpinned | Remote model revision; a commit SHA provides reproducibility |
| `EMBEDDING_MODEL_NAME` | `bge-m3` | Public model name exposed by the API |
| `EMBEDDING_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `EMBEDDING_USE_FP16` | `true` | Half precision on CUDA; CPU remains FP32 |
| `EMBEDDING_BATCH_SIZE` | `8` | Inference microbatch size |
| `EMBEDDING_MAX_BATCH_SIZE` | `32` | Maximum inputs per request; configurable up to 128 |
| `EMBEDDING_MAX_SEQ_LENGTH` | `8192` | Maximum tokens per input, including special tokens |
| `EMBEDDING_MAX_TOTAL_TOKENS` | `16384` | Maximum aggregate tokens per request |

`HF_HOME` controls the Hugging Face cache location outside the container default. Registry deployment requires `CADDY_DOMAIN` for HTTPS and optionally accepts `EMBEDDING_IMAGE` for an image tag or digest. Only source-build Compose reads `BIND_HOST` and `PORT`, defaulting to `127.0.0.1` and `8000`; these variables do not configure Uvicorn or registry-deployment ports.

Token and batch limits are admission controls, not memory guarantees. Long inputs can exhaust memory even at a microbatch size of one.

## Security

- Embedding and model-list endpoints require a Bearer key. Health and documentation endpoints are unauthenticated.
- The service is intended for trusted networks or deployment behind a TLS-terminating reverse proxy. EC2 ingress should be restricted to authorized clients or security groups.
- HTTP body-size limits and request-rate controls belong at the reverse proxy; application input validation occurs after JSON parsing.
- Runtime secrets are excluded from the Docker build context. `.env` is ignored by Git and must remain outside version control.

## Development and verification

The project targets Python 3.12–3.13 and uses uv with a committed dependency lockfile. The `cpu` and `cuda` extras are mutually exclusive; the base dependency set does not include model inference dependencies.

| Test layer | Coverage |
| --- | --- |
| API and engine unit tests | Authentication, validation, response contracts, output selection, token limits, device selection, and model-asset checks |
| HTTP client integration | Real LiteLLM and OpenAI SDK clients against the application, with model inference replaced by fixtures |
| CPU runtime integration | Real FlagEmbedding inference on tiny generated local weights; no Hugging Face model download |
| Container build check | FlagEmbedding import with the image's installed dependencies |

Reference verification commands:

```bash
uv run --frozen --group integration pytest -q
uv run --frozen --extra cpu --group integration pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

CPU runtime tests are skipped when model dependencies are absent. The suite does not establish full BGE-M3 embedding quality, production performance, or compatibility with a particular host's GPU driver.

## Scope

The service does not provide ColBERT vectors, document chunking, token-ID input, dimensionality reduction, vector storage, or retrieval-score fusion. Hybrid output supplies two representations; search and ranking remain the responsibility of the consuming application.
