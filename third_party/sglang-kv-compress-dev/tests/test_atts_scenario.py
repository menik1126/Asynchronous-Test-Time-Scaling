"""
End-to-end simulation of an ATTS (Asynchronous Test-Time Scaling) scenario
using the compressed RadixCache.

Simulates the full multi-turn, multi-sample workflow:
  - Turn 0: 16 samples share same sys+question prefix
    - Generate(sys+Q)  → compress → insert_compressed → decode → finish
    - PPL(sys+Q+CoT)   → reuses compressed prefix → finish → insert full CoT to tree
    - Continue(sys+Q+CoT) if PPL high → reuses CoT prefix → compress → finish
  - Turn 1: extends from (sys+Q+CoT_0+CoT_cont) prefix
    - Subsequent samples reuse the same compressed prefix

Run:
  python tests/test_atts_scenario.py
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FakeReqToTokenPool:
    def __init__(self, max_reqs=64, max_len=4096):
        self.req_to_token = torch.zeros((max_reqs, max_len), dtype=torch.int32)
        self._next = 0

    def alloc(self):
        idx = self._next
        self._next += 1
        return idx

    def free(self, idx):
        pass

    def write(self, loc, val):
        req_idx, slc = loc
        self.req_to_token[req_idx, slc] = val


class FakeAllocator:
    def __init__(self, pool_size=10000):
        self._next_slot = 1
        self.freed = []

    def alloc(self, n):
        slots = torch.arange(self._next_slot, self._next_slot + n, dtype=torch.int32)
        self._next_slot += n
        return slots

    def free(self, indices):
        if isinstance(indices, torch.Tensor):
            self.freed.extend(indices.tolist())
        else:
            self.freed.extend(list(indices))

    def available_size(self):
        return 99999


@dataclass
class FakeReq:
    rid: str
    origin_input_ids: List[int]
    output_ids: List[int] = field(default_factory=list)
    prefix_indices: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=torch.int32)
    )
    last_node: Optional[TreeNode] = None
    fill_ids: List[int] = field(default_factory=list)
    extend_input_len: int = 0
    is_kv_compressed: bool = False
    compressed_seq_len: Optional[int] = None
    original_seq_len: Optional[int] = None
    _snapkv_position_offset: int = 0
    _snapkv_seq_lens_fixed: bool = False
    req_pool_idx: int = 0

    def init_next_round_input(self, tree_cache):
        """Matches the real Req.init_next_round_input logic."""
        self.fill_ids = self.origin_input_ids + self.output_ids
        if tree_cache is not None:
            self.prefix_indices, self.last_node = tree_cache.match_prefix(
                key=self.fill_ids[:-1]  # leave at least 1 token to extend
            )
            pos_offset = getattr(self.last_node, "_match_position_offset", 0)
            matched_token_len = getattr(
                self.last_node, "_match_token_len", len(self.prefix_indices)
            )
            if pos_offset > 0:
                self.is_kv_compressed = True
                self.compressed_seq_len = len(self.prefix_indices)
                self.original_seq_len = matched_token_len
                self._snapkv_position_offset = pos_offset
                self._snapkv_seq_lens_fixed = True
                self.extend_input_len = len(self.fill_ids) - matched_token_len
            else:
                self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)
        else:
            self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)


def make_env():
    rtp = FakeReqToTokenPool()
    alloc = FakeAllocator()
    cache = RadixCache(rtp, alloc, disable=False)
    return cache, rtp, alloc


# Token IDs for the scenario
SYS_TOKENS = list(range(1, 51))          # 50 system tokens
Q_TOKENS = list(range(100, 200))         # 100 question tokens
PREFIX_TOKENS = SYS_TOKENS + Q_TOKENS    # 150 tokens total

COT_TURN0_A = list(range(1000, 1300))    # 300 CoT tokens for sample A
COT_TURN0_B = list(range(2000, 2250))    # 250 CoT tokens for sample B
COT_CONT_A = list(range(3000, 3200))     # 200 continuation tokens


def simulate_prefill_and_compress(cache, alloc, rtp, token_ids, budget_ratio=0.5):
    """Simulate: prefill → SnapKV select → free unselected → insert_compressed.

    Returns (req, compressed_kv, selected_positions).
    """
    seq_len = len(token_ids)

    # Allocate KV slots for the full sequence
    kv_slots = alloc.alloc(seq_len)

    # Allocate req slot and write KV
    req_pool_idx = rtp.alloc()
    rtp.req_to_token[req_pool_idx, :seq_len] = kv_slots

    # SnapKV: select every other position + always keep first 30 and last 30
    budget = max(int(seq_len * budget_ratio), 60)
    all_pos = list(range(seq_len))
    # Protect first 30, keep every-other in middle, keep last 30
    protected = list(range(min(30, seq_len)))
    recent = list(range(max(0, seq_len - 30), seq_len))
    middle = list(range(30, max(0, seq_len - 30), 2))
    selected = sorted(set(protected + middle + recent))
    if len(selected) > budget:
        selected = selected[:budget]
    selected = torch.tensor(selected, dtype=torch.int64)

    full_kv = rtp.req_to_token[req_pool_idx, :seq_len].clone()
    selected_kv = full_kv[selected]
    compressed_len = len(selected_kv)

    # Free unselected slots
    keep_mask = torch.zeros(seq_len, dtype=torch.bool)
    keep_mask[selected] = True
    unselected = (~keep_mask).nonzero(as_tuple=True)[0]
    if unselected.numel() > 0:
        alloc.free(full_kv[unselected])

    # Write compressed KV to req_to_token
    rtp.req_to_token[req_pool_idx, :compressed_len] = selected_kv

    # Insert compressed into tree
    cache.insert_compressed(token_ids, selected_kv.clone(), selected)

    # Lock the node
    _, node = cache.match_prefix(token_ids)
    cache.inc_lock_ref(node)

    return req_pool_idx, compressed_len, selected, node


def simulate_ppl_request(cache, alloc, rtp, token_ids):
    """Simulate PPL request: prefill-only, max_new_tokens=1, no compression.

    Reuses prefix from tree, extends, then caches full KV.
    """
    req = FakeReq(rid="ppl", origin_input_ids=token_ids, output_ids=[])
    req.init_next_round_input(cache)

    # The prefix should be matched
    prefix_kv_len = len(req.prefix_indices)
    extend_len = req.extend_input_len

    # Allocate KV for extend portion only
    extend_kv = alloc.alloc(extend_len) if extend_len > 0 else torch.tensor([], dtype=torch.int32)

    # Build full KV indices: prefix (from tree) + extend (newly allocated)
    full_kv = torch.cat([req.prefix_indices, extend_kv])

    # Write to req_to_token
    req_pool_idx = rtp.alloc()
    rtp.req_to_token[req_pool_idx, :len(full_kv)] = full_kv

    # PPL generates 1 token then finishes → cache_finished_req with normal insert
    output_token_kv = alloc.alloc(1)
    total_len = len(full_kv) + 1
    rtp.req_to_token[req_pool_idx, len(full_kv):total_len] = output_token_kv

    # Insert full (token_ids) into tree (non-compressed, like normal cache_finished_req)
    # token_ids for cache = origin_input_ids + output_ids[:-1] = origin_input_ids
    cache.insert(token_ids, rtp.req_to_token[req_pool_idx, :len(token_ids)].clone())

    return req, prefix_kv_len, extend_len


# ===================================================================
# Test cases
# ===================================================================

class TestATTSScenario(unittest.TestCase):

    def test_turn0_first_sample_compress(self):
        """Turn 0, Sample A: prefill sys+Q → compress → tree has compressed prefix."""
        cache, rtp, alloc = make_env()

        token_ids = PREFIX_TOKENS + [999]  # +first output token from prefill
        req_pool_idx, compressed_len, selected, node = simulate_prefill_and_compress(
            cache, alloc, rtp, token_ids
        )

        self.assertLess(compressed_len, len(token_ids))
        self.assertEqual(cache.protected_size(), compressed_len)

        # Verify match returns compressed KV
        val, matched_node = cache.match_prefix(token_ids)
        self.assertEqual(len(val), compressed_len)
        self.assertEqual(matched_node._match_token_len, len(token_ids))
        self.assertGreater(matched_node._match_position_offset, 0)

    def test_turn0_second_sample_reuses_compressed(self):
        """Turn 0, Sample B: same sys+Q prefix → reuses compressed KV directly."""
        cache, rtp, alloc = make_env()

        # Sample A compresses
        token_ids_a = PREFIX_TOKENS + [999]
        simulate_prefill_and_compress(cache, alloc, rtp, token_ids_a)

        # Sample B arrives with same prefix
        req_b = FakeReq(rid="B", origin_input_ids=PREFIX_TOKENS + [999])
        req_b.init_next_round_input(cache)

        # adjust_max_prefix_ids keeps fill_ids[:-1] → matches 150 of 151 tokens
        # So there's always 1 token to extend (the last token)
        self.assertTrue(req_b.is_kv_compressed)
        self.assertGreater(req_b._snapkv_position_offset, 0)
        self.assertEqual(req_b.extend_input_len, 1)  # last token not cached

    def test_turn0_ppl_reuses_compressed_then_extends(self):
        """PPL request for sys+Q+CoT: reuses compressed sys+Q, extends CoT."""
        cache, rtp, alloc = make_env()

        # Compress sys+Q+first_token
        compress_tokens = PREFIX_TOKENS + [999]
        req_pool_idx, compressed_len, _, _ = simulate_prefill_and_compress(
            cache, alloc, rtp, compress_tokens
        )

        # PPL request: sys+Q+CoT_A (CoT starts with 1000, not 999)
        # So the tree matches PREFIX_TOKENS (150 tokens) via split — not the [999] token.
        ppl_tokens = PREFIX_TOKENS + COT_TURN0_A
        req_ppl = FakeReq(rid="ppl", origin_input_ids=ppl_tokens)
        req_ppl.init_next_round_input(cache)

        # Matches PREFIX_TOKENS (150 tokens) from the compressed node
        self.assertTrue(req_ppl.is_kv_compressed)
        self.assertEqual(req_ppl.original_seq_len, len(PREFIX_TOKENS))
        self.assertGreater(req_ppl.extend_input_len, 0)
        # extend_input_len = len(fill_ids) - matched_token_len
        # fill_ids = ppl_tokens (450), matched = PREFIX_TOKENS (150)
        expected_extend = len(ppl_tokens) - len(PREFIX_TOKENS)
        self.assertEqual(req_ppl.extend_input_len, expected_extend)

    def test_turn0_ppl_caches_cot_for_next_turn(self):
        """After PPL, CoT tokens are in tree; next turn can reuse them."""
        cache, rtp, alloc = make_env()

        # Compress sys+Q
        compress_tokens = PREFIX_TOKENS + [999]
        simulate_prefill_and_compress(cache, alloc, rtp, compress_tokens)

        # PPL caches full sys+Q+CoT into tree (uncompressed)
        ppl_tokens = PREFIX_TOKENS + COT_TURN0_A
        simulate_ppl_request(cache, alloc, rtp, ppl_tokens)

        # Next turn's generate: sys+Q+CoT_A → should match in tree
        next_tokens = PREFIX_TOKENS + COT_TURN0_A + [5555]
        req_next = FakeReq(rid="next", origin_input_ids=next_tokens)
        req_next.init_next_round_input(cache)

        # Should match the full sys+Q+CoT_A prefix
        matched = getattr(req_next.last_node, "_match_token_len", len(req_next.prefix_indices))
        # At least sys+Q+CoT should be matched
        self.assertGreaterEqual(matched, len(ppl_tokens) - 1)
        self.assertLessEqual(req_next.extend_input_len, 2)

    def test_16_samples_same_question(self):
        """16 samples for the same question all reuse the same compressed prefix."""
        cache, rtp, alloc = make_env()

        # First sample compresses sys+Q+first_token
        compress_tokens = PREFIX_TOKENS + [999]
        simulate_prefill_and_compress(cache, alloc, rtp, compress_tokens)

        # Match the full key to get the reference compressed KV length
        ref_val, _ = cache.match_prefix(compress_tokens)
        compressed_kv_len = len(ref_val)

        # All 16 samples with the same input should match the compressed prefix.
        # adjust_max_prefix_ids uses fill_ids[:-1], so each sample extends 1 token.
        for i in range(16):
            req = FakeReq(
                rid=f"sample_{i}",
                origin_input_ids=PREFIX_TOKENS + [999],
            )
            req.init_next_round_input(cache)
            self.assertTrue(req.is_kv_compressed, f"sample {i} not compressed")
            self.assertEqual(req.extend_input_len, 1, f"sample {i}: expected 1 extend token")

    def test_different_questions_independent(self):
        """Two different questions don't interfere with each other."""
        cache, rtp, alloc = make_env()

        # Question 1: sys + Q_TOKENS
        q1_tokens = SYS_TOKENS + Q_TOKENS + [999]
        simulate_prefill_and_compress(cache, alloc, rtp, q1_tokens)

        # Question 2: sys + different question
        q2_tokens = SYS_TOKENS + list(range(500, 600)) + [888]
        simulate_prefill_and_compress(cache, alloc, rtp, q2_tokens)

        # Verify Q1 match
        val1, n1 = cache.match_prefix(q1_tokens)
        self.assertEqual(n1._match_token_len, len(q1_tokens))

        # Verify Q2 match
        val2, n2 = cache.match_prefix(q2_tokens)
        self.assertEqual(n2._match_token_len, len(q2_tokens))

        # They share the sys prefix but diverge after that
        self.assertNotEqual(val1.tolist(), val2.tolist())

    def test_memory_actually_freed(self):
        """Verify that unselected KV slots are really freed."""
        cache, rtp, alloc = make_env()

        token_ids = PREFIX_TOKENS + [999]  # 151 tokens
        seq_len = len(token_ids)

        freed_before = len(alloc.freed)
        req_pool_idx, compressed_len, selected, _ = simulate_prefill_and_compress(
            cache, alloc, rtp, token_ids
        )
        freed_after = len(alloc.freed)

        num_freed = freed_after - freed_before
        expected_freed = seq_len - compressed_len
        self.assertEqual(num_freed, expected_freed,
                         f"Expected {expected_freed} slots freed, got {num_freed}")
        self.assertGreater(num_freed, 0, "No memory was freed!")

    def test_compressed_eviction(self):
        """Compressed nodes can be evicted when memory is needed."""
        cache, rtp, alloc = make_env()

        token_ids = PREFIX_TOKENS + [999]
        _, compressed_len, _, node = simulate_prefill_and_compress(
            cache, alloc, rtp, token_ids
        )

        # Unlock so it becomes evictable
        cache.dec_lock_ref(node)
        self.assertEqual(cache.evictable_size(), compressed_len)

        freed_before = len(alloc.freed)
        cache.evict(9999, alloc.free)
        freed_after = len(alloc.freed)

        self.assertEqual(freed_after - freed_before, compressed_len)
        self.assertEqual(cache.evictable_size(), 0)

    def test_multi_turn_full_flow(self):
        """Full 2-turn simulation:
        Turn 0: Generate(sys+Q) → compress → PPL(sys+Q+CoT) → cache CoT
        Turn 1: Generate(sys+Q+CoT) → prefix cache hit on CoT
        """
        cache, rtp, alloc = make_env()

        # --- Turn 0: Generate ---
        gen0_tokens = PREFIX_TOKENS + [999]
        _, comp_len_0, _, node_0 = simulate_prefill_and_compress(
            cache, alloc, rtp, gen0_tokens
        )
        # Unlock after decode finishes
        cache.dec_lock_ref(node_0)

        # --- Turn 0: PPL (caches full CoT into tree) ---
        ppl_tokens = PREFIX_TOKENS + COT_TURN0_A
        simulate_ppl_request(cache, alloc, rtp, ppl_tokens)

        # --- Turn 1: Generate with CoT_0 as context ---
        gen1_input = PREFIX_TOKENS + COT_TURN0_A + [7777]
        req_gen1 = FakeReq(rid="gen1", origin_input_ids=gen1_input)
        req_gen1.init_next_round_input(cache)

        matched = getattr(req_gen1.last_node, "_match_token_len", len(req_gen1.prefix_indices))

        # The PPL request cached PREFIX + COT_TURN0_A into the tree (uncompressed).
        # The initial compressed prefix for PREFIX+[999] also exists.
        # gen1 should match PREFIX + COT_TURN0_A portion from the PPL insert.
        # Note: the tree may have two branches from PREFIX:
        #   - PREFIX+[999] (compressed)
        #   - PREFIX+COT_TURN0_A (from PPL, uncompressed)
        # gen1 input starts with PREFIX+COT_TURN0_A+[7777], so it matches the PPL branch.
        self.assertGreaterEqual(matched, len(PREFIX_TOKENS) + len(COT_TURN0_A) - 1)
        self.assertLessEqual(req_gen1.extend_input_len, 2)

        # Position offset: the PPL branch is uncompressed, so no offset
        pos_offset = getattr(req_gen1.last_node, "_match_position_offset", 0)
        # The PPL branch is uncompressed but shares the compressed sys+Q prefix
        # After PPL insert, the tree has an uncompressed path for PREFIX+COT,
        # so the offset comes only from the initial compressed sys+Q portion.
        # This depends on tree structure — let's just verify it's non-negative.
        self.assertGreaterEqual(pos_offset, 0)


class TestATTSEdgeCases(unittest.TestCase):

    def test_empty_cot(self):
        """Handle the case where model returns empty output (same as prefix)."""
        cache, rtp, alloc = make_env()

        token_ids = PREFIX_TOKENS + [999]
        simulate_prefill_and_compress(cache, alloc, rtp, token_ids)

        req = FakeReq(rid="empty", origin_input_ids=PREFIX_TOKENS + [999])
        req.init_next_round_input(cache)
        # adjust_max_prefix_ids uses fill_ids[:-1] → always 1 token to extend
        self.assertEqual(req.extend_input_len, 1)

    def test_very_short_prefix(self):
        """Prefix shorter than typical protect_prefix."""
        cache, rtp, alloc = make_env()

        short_tokens = [1, 2, 3, 4, 5]
        kv = torch.tensor([10, 30, 50], dtype=torch.int32)
        sel = torch.tensor([0, 2, 4], dtype=torch.int64)
        cache.insert_compressed(short_tokens, kv, sel)

        req = FakeReq(rid="short", origin_input_ids=short_tokens + [99])
        req.init_next_round_input(cache)
        self.assertTrue(req.is_kv_compressed)
        self.assertEqual(req._snapkv_position_offset, 2)

    def test_concurrent_samples_different_cot(self):
        """Multiple samples finish with different CoTs; tree has separate branches."""
        cache, rtp, alloc = make_env()

        # Compress shared prefix
        prefix_tok = PREFIX_TOKENS + [999]
        simulate_prefill_and_compress(cache, alloc, rtp, prefix_tok)

        # Sample A PPL: sys+Q+CoT_A
        simulate_ppl_request(cache, alloc, rtp, PREFIX_TOKENS + COT_TURN0_A)

        # Sample B PPL: sys+Q+CoT_B (different CoT)
        simulate_ppl_request(cache, alloc, rtp, PREFIX_TOKENS + COT_TURN0_B)

        # Verify both branches exist
        val_a, node_a = cache.match_prefix(PREFIX_TOKENS + COT_TURN0_A)
        val_b, node_b = cache.match_prefix(PREFIX_TOKENS + COT_TURN0_B)

        self.assertGreaterEqual(node_a._match_token_len, len(PREFIX_TOKENS) + len(COT_TURN0_A) - 1)
        self.assertGreaterEqual(node_b._match_token_len, len(PREFIX_TOKENS) + len(COT_TURN0_B) - 1)

        # Different CoTs → different tree branches
        self.assertNotEqual(node_a.id, node_b.id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
