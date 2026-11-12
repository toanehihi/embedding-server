"""Optional real FlagEmbedding inference on tiny locally generated weights, no HF downloads."""

import math

import pytest

pytest.importorskip("FlagEmbedding")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.model_runtime


@pytest.fixture(scope="module")
def local_engine(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

    from app.config import Settings
    from app.engine import BGEEngine

    path = tmp_path_factory.mktemp("tiny-bge-m3")
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "hello": 4, "world": 5}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=32,
    ).save_pretrained(path)
    config = XLMRobertaConfig(
        vocab_size=6,
        hidden_size=1024,
        num_hidden_layers=1,
        num_attention_heads=8,
        intermediate_size=32,
        max_position_embeddings=34,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    XLMRobertaModel(config).save_pretrained(path)
    sparse = torch.nn.Linear(1024, 1)
    # A known trained-head stand-in: every lexical token has weight 1,
    # repeated IDs are max-pooled, and special tokens must be excluded.
    with torch.no_grad():
        sparse.weight.zero_()
        sparse.bias.fill_(1.0)
    torch.save(sparse.state_dict(), path / "sparse_linear.pt")
    torch.save(torch.nn.Linear(1024, 1024).state_dict(), path / "colbert_linear.pt")
    return BGEEngine(
        Settings(
            api_key="local-runtime-test-key",
            model_id=str(path),
            model_revision=None,
            device="cpu",
            batch_size=2,
            max_seq_length=32,
            _env_file=None,
        )
    )


@pytest.mark.parametrize("output_type", ["dense", "sparse", "hybrid"])
def test_real_cpu_encode_heads_and_token_accounting(local_engine, output_type):
    result = local_engine.embed(["hello hello world", "world"], output_type=output_type)
    assert result.prompt_tokens == 8  # (3 lexical + 2 special) + (1 lexical + 2 special)
    if output_type == "sparse":
        assert result.vectors is None
    else:
        assert len(result.vectors) == 2
        for vector in result.vectors:
            assert len(vector) == 1024
            assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0, abs=1e-5)
    if output_type == "dense":
        assert result.sparse_vectors is None
    else:
        first, second = result.sparse_vectors
        assert first.indices == [4, 5]
        assert first.values == [1.0, 1.0]
        assert second.indices == [5]
        assert second.values == [1.0]
