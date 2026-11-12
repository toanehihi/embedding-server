import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)


class InputTooLong(ValueError):
    pass


@dataclass
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@dataclass
class EmbeddingResult:
    vectors: list[list[float]] | None
    prompt_tokens: int
    sparse_vectors: list[SparseEmbedding] | None = None


def resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but unavailable. Check NVIDIA driver and PyTorch build.")
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested


class BGEEngine:
    def __init__(self, settings: Settings, *, model=None):
        self.settings = settings
        self.device = "cpu"
        if model is None:
            import torch
            from FlagEmbedding import BGEM3FlagModel

            self.device = resolve_device(settings.device, torch.cuda.is_available())
            model_path = settings.model_id
            if not Path(model_path).is_dir():
                from huggingface_hub import snapshot_download

                # FlagEmbedding does not forward revision to all assets. Load one pinned
                # snapshot so dense weights, learned sparse head and tokenizer stay aligned.
                model_path = snapshot_download(
                    repo_id=settings.model_id,
                    revision=settings.model_revision,
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.pt", "*.txt"],
                )
            # FlagEmbedding otherwise silently initializes random heads if either file
            # is absent. Never serve meaningless sparse weights from an incomplete cache.
            required_heads = ("sparse_linear.pt", "colbert_linear.pt")
            if any(not (Path(model_path) / name).is_file() for name in required_heads):
                raise RuntimeError(
                    "BGE-M3 trained auxiliary heads missing: sparse_linear.pt and "
                    "colbert_linear.pt are both required. Use a complete BAAI/bge-m3 snapshot."
                )
            model = BGEM3FlagModel(
                model_path,
                devices="cuda:0" if self.device == "cuda" else "cpu",
                use_fp16=self.device == "cuda" and settings.use_fp16,
                normalize_embeddings=True,
                trust_remote_code=False,
            )
            logger.info("Loaded %s on %s", settings.model_id, self.device)
        self.model = model

    def embed(self, texts: list[str], output_type: str = "dense") -> EmbeddingResult:
        if output_type not in {"dense", "sparse", "hybrid"}:
            raise ValueError("output_type must be dense, sparse, or hybrid")
        tokens = self.model.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        lengths = [len(ids) for ids in tokens]
        if max(lengths) > self.settings.max_seq_length:
            raise InputTooLong(
                f"Each input must be at most {self.settings.max_seq_length} tokens "
                "including special tokens; split long documents into chunks."
            )
        token_count = sum(lengths)
        if token_count > self.settings.max_total_tokens:
            raise InputTooLong(
                f"Request exceeds {self.settings.max_total_tokens} total tokens; use smaller batches."
            )
        output = self.model.encode(
            texts,
            batch_size=self.settings.batch_size,
            max_length=self.settings.max_seq_length,
            return_dense=output_type != "sparse",
            return_sparse=output_type != "dense",
            return_colbert_vecs=False,
        )
        vectors = output["dense_vecs"].tolist() if output_type != "sparse" else None
        sparse_vectors = None
        if output_type != "dense":
            sparse_vectors = []
            for weights in output["lexical_weights"]:
                pairs = sorted(
                    (int(token_id), float(weight)) for token_id, weight in weights.items()
                )
                sparse_vectors.append(
                    SparseEmbedding(
                        indices=[token_id for token_id, _ in pairs],
                        values=[weight for _, weight in pairs],
                    )
                )
        return EmbeddingResult(
            vectors=vectors, prompt_tokens=token_count, sparse_vectors=sparse_vectors
        )
