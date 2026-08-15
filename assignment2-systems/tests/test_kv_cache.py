import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.kv_cache.cache import CachedTransformerLM, generate_greedy_no_cache


def _make_model():
    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=32,
        context_length=16,
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_ff=32,
        rope_theta=10_000.0,
    )
    model.eval()
    return model


def test_kv_cache_prefill_matches_original_forward():
    model = _make_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    cached_model = CachedTransformerLM(model)

    logits_cached, cache = cached_model.forward(input_ids)
    logits_reference = model(input_ids)

    torch.testing.assert_close(logits_cached, logits_reference, rtol=1e-5, atol=1e-5)
    assert cache.seq_len() == input_ids.shape[-1]
    for key_cache, value_cache in zip(cache.keys, cache.values, strict=True):
        assert key_cache is not None
        assert value_cache is not None
        assert key_cache.shape[-2] == input_ids.shape[-1]
        assert value_cache.shape[-2] == input_ids.shape[-1]


def test_kv_cache_incremental_logits_match_full_prefix():
    model = _make_model()
    cached_model = CachedTransformerLM(model)
    prompt = torch.tensor([[1, 2, 3, 4]])
    suffix = torch.tensor([[5, 6, 7]])

    _logits, cache = cached_model.forward(prompt)
    prefix = prompt
    for token in suffix.split(1, dim=-1):
        logits_cached, cache = cached_model.forward(token, cache)
        prefix = torch.cat((prefix, token), dim=-1)
        logits_reference = model(prefix)
        torch.testing.assert_close(logits_cached[:, -1], logits_reference[:, -1], rtol=1e-5, atol=1e-5)
        assert cache.seq_len() == prefix.shape[-1]


def test_kv_cache_greedy_generation_matches_no_cache():
    model = _make_model()
    prompt = torch.tensor([[3, 1, 4]])

    cached_tokens = CachedTransformerLM(model).generate_greedy(prompt, max_new_tokens=5)
    reference_tokens = generate_greedy_no_cache(model, prompt, max_new_tokens=5)

    torch.testing.assert_close(cached_tokens, reference_tokens)
