from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor


@dataclass
class KVCache:
    """Per-layer key/value cache for autoregressive inference."""

    keys: list[Tensor | None]
    values: list[Tensor | None]
    lengths: list[int] | None = None

    @classmethod
    def empty(cls, num_layers: int) -> KVCache:
        return cls(keys=[None] * num_layers, values=[None] * num_layers)

    @classmethod
    def preallocate(
        cls,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> KVCache:
        shape = (batch_size, num_heads, max_seq_len, head_dim)
        return cls(
            keys=[torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)],
            values=[torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)],
            lengths=[0] * num_layers,
        )

    def seq_len(self, layer_idx: int = 0) -> int:
        if self.lengths is not None:
            return self.lengths[layer_idx]
        key_cache = self.keys[layer_idx]
        return 0 if key_cache is None else key_cache.shape[-2]

    def append(self, layer_idx: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        key_cache = self.keys[layer_idx]
        value_cache = self.values[layer_idx]
        if self.lengths is not None:
            if key_cache is None or value_cache is None:
                raise ValueError("Preallocated cache tensors must not be None.")
            start = self.lengths[layer_idx]
            end = start + key.shape[-2]
            if end > key_cache.shape[-2]:
                raise ValueError(f"KV cache capacity exceeded: requested {end}, capacity {key_cache.shape[-2]}.")
            key_cache[..., start:end, :] = key
            value_cache[..., start:end, :] = value
            self.lengths[layer_idx] = end
            return key_cache[..., :end, :], value_cache[..., :end, :]

        if key_cache is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            self.keys[layer_idx] = torch.cat((key_cache, key), dim=-2)
            self.values[layer_idx] = torch.cat((value_cache, value), dim=-2)
        return self.keys[layer_idx], self.values[layer_idx]


def _causal_mask(query_positions: Tensor, key_len: int) -> Tensor:
    key_positions = torch.arange(key_len, device=query_positions.device)
    return query_positions[..., :, None] >= key_positions[None, None, :]


def _attention(query: Tensor, key: Tensor, value: Tensor, query_positions: Tensor) -> Tensor:
    scale = key.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if query.shape[-2] > 1:
        mask = _causal_mask(query_positions, key.shape[-2]).unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value)


class CachedTransformerLM:
    """A thin cached-inference wrapper around the A2 BasicsTransformerLM.

    The wrapper reuses the original model weights and modules, but owns the KV
    cache data flow so the baseline model code remains untouched.
    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def forward(self, input_ids: Tensor, cache: KVCache | None = None) -> tuple[Tensor, KVCache]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if cache is None:
            cache = KVCache.empty(len(self.model.layers))

        hidden = self.model.token_embeddings(input_ids)
        batch_size, seq_len = input_ids.shape
        start_pos = cache.seq_len(0)
        positions = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device).expand(batch_size, seq_len)

        for layer_idx, layer in enumerate(self.model.layers):
            hidden = self._forward_layer(layer, layer_idx, hidden, positions, cache)

        hidden = self.model.ln_final(hidden)
        logits = self.model.lm_head(hidden)
        return logits, cache

    def _forward_layer(self, layer, layer_idx: int, x: Tensor, positions: Tensor, cache: KVCache) -> Tensor:
        attn_output = self._forward_attention(layer.attn, layer_idx, layer.ln1(x), positions, cache)
        x = x + attn_output
        return x + layer.ffn(layer.ln2(x))

    def _forward_attention(self, attn, layer_idx: int, x: Tensor, positions: Tensor, cache: KVCache) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        if d_model != attn.d_model:
            raise ValueError(f"Expected d_model={attn.d_model}, got {d_model}.")

        query = rearrange(attn.q_proj(x), "batch seq (heads d) -> batch heads seq d", heads=attn.num_heads)
        key = rearrange(attn.k_proj(x), "batch seq (heads d) -> batch heads seq d", heads=attn.num_heads)
        value = rearrange(attn.v_proj(x), "batch seq (heads d) -> batch heads seq d", heads=attn.num_heads)

        if attn.positional_encoder is not None:
            rope_positions = positions[:, None, :]
            query = attn.positional_encoder(query, rope_positions)
            key = attn.positional_encoder(key, rope_positions)

        cached_key, cached_value = cache.append(layer_idx, key, value)
        attn_output = _attention(query, cached_key, cached_value, positions)
        attn_output = rearrange(attn_output, "batch heads seq d -> batch seq (heads d)").contiguous()
        return attn.output_proj(attn_output)

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids
        first_layer_attn = self.model.layers[0].attn
        max_seq_len = min(self.model.context_length, input_ids.shape[-1] + max_new_tokens)
        cache = KVCache.preallocate(
            num_layers=len(self.model.layers),
            batch_size=input_ids.shape[0],
            num_heads=first_layer_attn.num_heads,
            max_seq_len=max_seq_len,
            head_dim=first_layer_attn.d_k,
            dtype=self.model.token_embeddings.weight.dtype,
            device=input_ids.device,
        )
        logits, cache = self.forward(input_ids, cache)
        for step in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            generated = torch.cat((generated, next_token), dim=-1)
            if step + 1 < max_new_tokens:
                logits, cache = self.forward(next_token, cache)
        return generated[:, input_ids.shape[-1] :]


@torch.no_grad()
def generate_with_kv_cache(
    model,
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Tensor:
    return CachedTransformerLM(model).generate_greedy(input_ids, max_new_tokens, eos_token_id=eos_token_id)


@torch.no_grad()
def generate_greedy_no_cache(model, input_ids: Tensor, max_new_tokens: int, eos_token_id: int | None = None) -> Tensor:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    generated = input_ids
    original_length = input_ids.shape[-1]
    for _ in range(max_new_tokens):
        context = generated[:, -model.context_length :]
        logits = model(context)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        generated = torch.cat((generated, next_token), dim=-1)
    return generated[:, original_length:]
