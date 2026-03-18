"""
SnapKV-style KV cache compression for SGLang.

Selects important tokens based on attention patterns in an observation window,
keeping those plus recent tokens. Works with the radix tree prefix cache:
only frees KV slots from the request's private portion, never from shared
prefix slots owned by the radix tree.

Reference: SnapKV: LLM Knows What You are Looking for Before Generation
           (Li et al., 2024)
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_obs_attn_scores(
    req,
    req_to_token_pool,
    kvcache,
    obs_window: int = 32,
    layer_ids: Optional[List[int]] = None,
    cached_q: Optional[dict] = None,
    req_index: int = 0,
    kv_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Compute attention scores from the last obs_window tokens to all tokens.

    If cached_q is provided (from prefill's RadixAttention forward), uses the
    real Q vectors for accurate scoring. Otherwise falls back to using K as
    a proxy for Q.

    Args:
        req: The request object with req_pool_idx.
        req_to_token_pool: Maps req to KV pool indices.
        kvcache: The KV cache (MHATokenToKVPool or similar).
        obs_window: Number of recent tokens to use as the observation window.
        layer_ids: Which layers to aggregate scores from. If None, uses
            the last 1/4 of layers.
        cached_q: Dict mapping layer_id -> list of per-request Q tensors
            (from RadixAttention._cache_obs_q). Each tensor has shape
            [obs_tokens, num_q_heads, head_dim].
        req_index: Index of the request in the batch (for cached_q lookup).
        kv_seq_len: Override for the number of valid KV entries. When the
            request matched a compressed prefix, the KV count is less than
            the token count. If None, falls back to token-space length.

    Returns:
        Attention weights tensor of shape [num_kv_heads, obs_window, seq_len].
    """
    if kv_seq_len is not None:
        seq_len = kv_seq_len
    else:
        seq_len = len(req.origin_input_ids) + len(req.output_ids)
    if seq_len <= obs_window:
        return None

    kv_indices = req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len]

    if layer_ids is None:
        total_layers = kvcache.layer_num
        start = total_layers * 3 // 4
        layer_ids = list(range(start, total_layers))

    use_real_q = cached_q is not None and any(lid in cached_q for lid in layer_ids)

    all_attn = []
    for layer_id in layer_ids:
        k_buffer = kvcache.get_key_buffer(layer_id)
        k_seq = k_buffer[kv_indices]  # [seq_len, num_kv_heads, head_dim]
        num_kv_heads = k_seq.shape[1]
        head_dim = k_seq.shape[2]

        k = k_seq.permute(1, 0, 2)  # [num_kv_heads, seq_len, head_dim]

        if use_real_q and layer_id in cached_q:
            q_raw = cached_q[layer_id][req_index]  # [obs, num_q_heads, head_dim]
            obs_len = min(q_raw.shape[0], obs_window)
            q_raw = q_raw[-obs_len:]
            num_q_heads = q_raw.shape[1]

            # GQA: average Q heads within each KV head group
            if num_q_heads != num_kv_heads:
                gqa_ratio = num_q_heads // num_kv_heads
                q_grouped = q_raw.view(obs_len, num_kv_heads, gqa_ratio, head_dim)
                q_avg = q_grouped.mean(dim=2)  # [obs, num_kv_heads, head_dim]
            else:
                q_avg = q_raw

            q = q_avg.permute(1, 0, 2)  # [num_kv_heads, obs_len, head_dim]
        else:
            # Fallback: use K as proxy for Q
            k_obs = k_seq[seq_len - obs_window:]
            q = k_obs.permute(1, 0, 2)  # [num_kv_heads, obs_window, head_dim]

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1)
        all_attn.append(attn)

    attn_weights = torch.stack(all_attn).mean(dim=0)
    return attn_weights


def snapkv_select_positions(
    attn_weights: torch.Tensor,
    seq_len: int,
    budget: int = 256,
    recent_window: int = 64,
    obs_window: int = 32,
    kernel_size: int = 5,
    protect_prefix: int = 0,
) -> torch.Tensor:
    """Select important token positions using SnapKV attention pooling.

    Args:
        attn_weights: Attention weights from the observation window.
            Shape: [num_heads, obs_window, seq_len] or [obs_window, seq_len].
        seq_len: Total sequence length.
        budget: Number of important tokens to keep from the prefix region.
        recent_window: Number of recent tokens to always keep.
        obs_window: Size of the observation window (last N tokens before recent).
        kernel_size: Pooling kernel size for smoothing attention scores.
        protect_prefix: Number of tokens at the start to always keep (e.g., question tokens).
            These are excluded from the selection budget — they are kept unconditionally.

    Returns:
        Sorted tensor of selected position indices (on the same device as attn_weights).
    """
    if seq_len <= budget + recent_window + obs_window + protect_prefix:
        return torch.arange(seq_len, device=attn_weights.device)

    if attn_weights.dim() == 2:
        attn_weights = attn_weights.unsqueeze(0)

    prefix_len = seq_len - obs_window - recent_window

    if prefix_len <= budget + protect_prefix:
        return torch.arange(seq_len, device=attn_weights.device)

    # Protected prefix: always kept, not part of the selection pool
    protected = torch.arange(protect_prefix, device=attn_weights.device)

    # Only score the compressible region (after protected prefix, before obs+recent)
    compress_attn = attn_weights[:, :, protect_prefix:prefix_len]

    avg_attn = compress_attn.mean(dim=0)  # [obs_window, compressible_len]
    scores = avg_attn.sum(dim=0)  # [compressible_len]

    compressible_len = prefix_len - protect_prefix
    if kernel_size > 1 and compressible_len >= kernel_size:
        scores_1d = scores.unsqueeze(0).unsqueeze(0)
        scores_pooled = F.avg_pool1d(
            scores_1d,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        ).squeeze(0).squeeze(0)
        scores = scores_pooled[:compressible_len]

    actual_budget = min(budget, compressible_len)
    _, topk_idx = scores.topk(actual_budget)
    important = (topk_idx + protect_prefix).sort().values  # offset back to global positions

    obs = torch.arange(prefix_len, prefix_len + obs_window, device=attn_weights.device)
    recent = torch.arange(seq_len - recent_window, seq_len, device=attn_weights.device)
    selected = torch.cat([protected, important, obs, recent]).unique().sort().values

    return selected


def compress_request_kv(
    req,
    token_to_kv_pool_allocator,
    req_to_token_pool,
    selected_positions: torch.Tensor,
    prefix_len: int,
):
    """Compress a request's KV cache by freeing unselected private KV slots.

    Only frees slots from the request's private portion (after prefix_len).
    Shared prefix slots owned by the radix tree are never freed.

    Args:
        req: The request object (must have req_pool_idx, etc.).
        token_to_kv_pool_allocator: The KV pool allocator.
        req_to_token_pool: The request-to-token mapping pool.
        selected_positions: Sorted tensor of positions to keep.
        prefix_len: Length of the shared prefix (from radix tree match).

    Returns:
        Number of KV slots freed.
    """
    if getattr(req, "is_kv_compressed", False):
        return 0

    seq_len = len(req.origin_input_ids) + len(req.output_ids)
    full_kv_indices = req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len].clone()

    all_positions = torch.arange(seq_len, device=full_kv_indices.device)
    keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=full_kv_indices.device)
    keep_mask[selected_positions.to(full_kv_indices.device)] = True
    unselected = all_positions[~keep_mask]

    # Only free private (non-prefix) unselected slots
    private_unselected = unselected[unselected >= prefix_len]

    if private_unselected.numel() == 0:
        return 0

    slots_to_free = full_kv_indices[private_unselected]
    token_to_kv_pool_allocator.free(slots_to_free)

    # Compact selected KV indices into req_to_token
    selected_kv = full_kv_indices[selected_positions.to(full_kv_indices.device)]
    compressed_len = len(selected_kv)
    req_to_token_pool.req_to_token[req.req_pool_idx, :compressed_len] = selected_kv

    # Update request compression state
    req.is_kv_compressed = True
    req.compressed_seq_len = compressed_len
    req.original_seq_len = seq_len
    req.selected_positions = selected_positions.cpu()
    req.compress_prefix_len = prefix_len

    num_freed = private_unselected.numel()
    logger.debug(
        f"Compressed req {req.rid}: {seq_len} -> {compressed_len} tokens, "
        f"freed {num_freed} KV slots (prefix={prefix_len})"
    )
    return num_freed


def fork_compressed_kv(
    parent_req,
    child_req,
    token_to_kv_pool_allocator,
    req_to_token_pool,
    kvcache,
    clone: bool = True,
):
    """Transfer compressed KV from parent to child request (for ATTS fork).

    Args:
        parent_req: The parent request with compressed KV.
        child_req: The new forked request (must already have req_pool_idx allocated).
        token_to_kv_pool_allocator: The KV pool allocator.
        req_to_token_pool: The request-to-token mapping pool.
        kvcache: The KV cache object (MHATokenToKVPool or similar).
        clone: If True, copy KV data to new slots (safe, decoupled lifetime).
               If False, share the same slots (zero-copy, but must manage lifetime).

    Returns:
        The compressed_seq_len of the child.
    """
    if not parent_req.is_kv_compressed:
        raise ValueError("Parent request is not compressed")

    compressed_len = parent_req.compressed_seq_len

    if clone:
        new_slots = token_to_kv_pool_allocator.alloc(compressed_len)
        if new_slots is None:
            raise RuntimeError(
                f"Cannot allocate {compressed_len} slots for fork clone"
            )

        old_slots = req_to_token_pool.req_to_token[
            parent_req.req_pool_idx, :compressed_len
        ]

        for layer_id in range(kvcache.layer_num):
            kvcache.k_buffer[layer_id][new_slots] = kvcache.k_buffer[layer_id][old_slots]
            kvcache.v_buffer[layer_id][new_slots] = kvcache.v_buffer[layer_id][old_slots]

        req_to_token_pool.req_to_token[
            child_req.req_pool_idx, :compressed_len
        ] = new_slots
    else:
        old_slots = req_to_token_pool.req_to_token[
            parent_req.req_pool_idx, :compressed_len
        ]
        req_to_token_pool.req_to_token[
            child_req.req_pool_idx, :compressed_len
        ] = old_slots.clone()

    child_req.is_kv_compressed = True
    child_req.compressed_seq_len = compressed_len
    child_req.original_seq_len = parent_req.original_seq_len
    child_req.selected_positions = parent_req.selected_positions.clone()
    child_req.compress_prefix_len = parent_req.compress_prefix_len

    return compressed_len
