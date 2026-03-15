# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Radix attention."""

from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.k_scale = None
        self.v_scale = None

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)

        # Cache Q vectors for SnapKV attention scoring (only during extend/prefill)
        if (
            forward_batch.forward_mode.is_extend()
            and getattr(forward_batch, "snapkv_cache_q_layer_ids", None) is not None
            and self.layer_id in forward_batch.snapkv_cache_q_layer_ids
        ):
            self._cache_obs_q(q, forward_batch)

        return forward_batch.attn_backend.forward(
            q, k, v, self, forward_batch, save_kv_cache
        )

    def _cache_obs_q(self, q, forward_batch: ForwardBatch):
        """Cache the last obs_window Q vectors per request for SnapKV scoring."""
        import torch
        obs_window = forward_batch.snapkv_obs_window
        q_3d = q.view(-1, self.tp_q_head_num, self.qk_head_dim)

        if not hasattr(forward_batch, "snapkv_q_cache"):
            forward_batch.snapkv_q_cache = {}

        extend_start_loc = forward_batch.extend_start_loc
        extend_lens = forward_batch.extend_seq_lens - forward_batch.extend_prefix_lens

        per_req_q = []
        for i in range(forward_batch.batch_size):
            start = extend_start_loc[i].item()
            length = extend_lens[i].item()
            end = start + length
            take = min(obs_window, length)
            per_req_q.append(q_3d[end - take : end])  # [take, num_q_heads, head_dim]

        forward_batch.snapkv_q_cache[self.layer_id] = per_req_q
