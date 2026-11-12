import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [
        ("auto", False, "cpu"),
        ("auto", True, "cuda"),
        ("cpu", True, "cpu"),
        ("cuda", True, "cuda"),
    ],
)
def test_device_selection(requested, available, expected):
    from app.engine import resolve_device

    assert resolve_device(requested, available) == expected


def test_explicit_cuda_fails_without_gpu():
    from app.engine import resolve_device

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda", False)


class FakeModel:
    max_seq_length = 8192

    def tokenize_texts(self, texts, **kwargs):
        # Include special tokens; one-character text => 3 tokens.
        return {"input_ids": [[0] + [1] * len(text) + [2] for text in texts]}

    @property
    def tokenizer(self):
        return self.tokenize_texts

    def encode(
        self, texts, *, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs
    ):
        assert not return_colbert_vecs
        assert max_length == 8192
        assert batch_size == 2

        class Array:
            def tolist(self):
                return [[1.0] + [0.0] * 1023 for _ in texts]

        return {
            "dense_vecs": Array() if return_dense else None,
            "lexical_weights": [{"456": 0.75, "123": 0.25}, {}] if return_sparse else None,
            "colbert_vecs": None,
        }


@pytest.mark.parametrize(
    ("cuda_available", "fp16_requested", "expected_device", "expected_fp16"),
    [
        (False, True, "cpu", False),
        (True, True, "cuda", True),
        (True, False, "cuda", False),
    ],
)
def test_loader_uses_flagembedding_on_one_device_and_cpu_float32(
    monkeypatch, tmp_path, cuda_available, fp16_requested, expected_device, expected_fp16
):
    from app.config import Settings
    from app.engine import BGEEngine

    class LoadedModel(FakeModel):
        def __init__(
            self, model_name_or_path, *, devices, use_fp16, normalize_embeddings, **kwargs
        ):
            assert model_name_or_path == str(tmp_path)
            # Never let FlagEmbedding auto-select all GPUs and start a process pool.
            target = "cuda:0" if expected_device == "cuda" else "cpu"
            assert devices in (target, [target])
            assert use_fp16 is expected_fp16
            assert normalize_embeddings is True

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available)),
    )
    (tmp_path / "sparse_linear.pt").touch()
    (tmp_path / "colbert_linear.pt").touch()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: str(tmp_path)),
    )
    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(BGEM3FlagModel=LoadedModel))

    def reject_old_backend(*args, **kwargs):
        pytest.fail("Loader must use FlagEmbedding to load the learned sparse head")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=reject_old_backend),
    )
    settings = Settings(
        api_key="test-key-long-enough",
        device="auto",
        use_fp16=fp16_requested,
        model_revision=None,
        _env_file=None,
    )
    engine = BGEEngine(settings)
    assert engine.device == expected_device


def test_loader_pins_all_weights_and_tokenizer_to_snapshot(monkeypatch, tmp_path):
    from app.config import Settings
    from app.engine import BGEEngine

    class SnapshotModel(FakeModel):
        def __init__(self, model_name_or_path, **kwargs):
            # FlagEmbedding itself does not forward revision to all model assets.
            assert model_name_or_path == str(tmp_path)

    def download(*, repo_id, revision, **kwargs):
        assert repo_id == "BAAI/bge-m3"
        assert revision == "test-commit"
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    (tmp_path / "sparse_linear.pt").touch()
    (tmp_path / "colbert_linear.pt").touch()
    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(BGEM3FlagModel=SnapshotModel))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    settings = Settings(
        api_key="test-key-long-enough", model_revision="test-commit", _env_file=None
    )
    engine = BGEEngine(settings)
    assert engine.device == "cpu"


@pytest.mark.parametrize("present_heads", [[], ["sparse_linear.pt"], ["colbert_linear.pt"]])
def test_loader_rejects_incomplete_heads_instead_of_random_sparse(
    monkeypatch, tmp_path, present_heads
):
    from app.config import Settings
    from app.engine import BGEEngine

    for name in present_heads:
        (tmp_path / name).touch()

    def load_random_heads(*args, **kwargs):
        pytest.fail(
            "Must reject missing trained heads before FlagEmbedding initializes random ones"
        )

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", SimpleNamespace(BGEM3FlagModel=load_random_heads)
    )
    settings = Settings(api_key="test-key-long-enough", model_id=str(tmp_path), _env_file=None)
    with pytest.raises(RuntimeError, match="trained.*head"):
        BGEEngine(settings)


def test_engine_counts_real_tokenizer_tokens_and_returns_vectors():
    from app.config import Settings
    from app.engine import BGEEngine

    engine = BGEEngine(Settings(api_key="test-key-long-enough", batch_size=2), model=FakeModel())
    result = engine.embed(["a", "bc"])
    assert result.prompt_tokens == 7
    assert len(result.vectors) == 2
    assert result.vectors[0] == [1.0] + [0.0] * 1023


@pytest.mark.parametrize("output_type", ["dense", "sparse", "hybrid"])
def test_engine_selects_requested_heads_and_converts_sparse_ids(output_type):
    from app.config import Settings
    from app.engine import BGEEngine

    class SelectiveModel(FakeModel):
        def encode(self, texts, **kwargs):
            # Fails if sparse-only still computes dense, or dense-only computes sparse.
            assert kwargs["return_dense"] == (output_type != "sparse")
            assert kwargs["return_sparse"] == (output_type != "dense")
            return super().encode(texts, **kwargs)

    engine = BGEEngine(
        Settings(api_key="test-key-long-enough", batch_size=2), model=SelectiveModel()
    )
    result = engine.embed(["a", "bc"], output_type=output_type)
    assert result.prompt_tokens == 7
    if output_type == "sparse":
        assert result.vectors is None
    else:
        assert result.vectors == [[1.0] + [0.0] * 1023] * 2
    if output_type == "dense":
        assert result.sparse_vectors is None
    else:
        first, second = result.sparse_vectors
        assert first.indices == [123, 456]
        assert first.values == [0.25, 0.75]
        assert second.indices == []
        assert second.values == []


def test_engine_rejects_unknown_output_type():
    from app.config import Settings
    from app.engine import BGEEngine

    engine = BGEEngine(Settings(api_key="test-key-long-enough"), model=FakeModel())
    with pytest.raises(ValueError, match="output_type"):
        engine.embed(["a"], output_type="wrong")


@pytest.mark.parametrize(
    ("texts", "seq_limit", "token_budget"),
    [
        (["abc"], 4, 20),
        (["a", "b"], 4, 5),
    ],
)
def test_engine_rejects_token_overflow_instead_of_truncating(texts, seq_limit, token_budget):
    from app.config import Settings
    from app.engine import BGEEngine, InputTooLong

    settings = Settings(
        api_key="test-key-long-enough", max_seq_length=seq_limit, max_total_tokens=token_budget
    )
    engine = BGEEngine(settings, model=FakeModel())
    with pytest.raises(InputTooLong):
        engine.embed(texts)
