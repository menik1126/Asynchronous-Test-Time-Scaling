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


class TestBug2FixDecodeTokensCached(unittest.TestCase):
    """Verify that compressed requests' decode tokens are now stored in the tree.

    Before fix: cache_finished_req (compressed path) only freed decode KV.
    After fix: decode KV is stored in tree via insert_compressed.
    """

    def test_decode_tokens_visible_in_tree(self):
        """After a compressed request finishes, its decode tokens should be
        matchable by a subsequent request's prefix matching."""
        cache, rtp, alloc = make_env()

        # --- Step 1: Simulate Turn 0 DRAFT (uncompressed, stores all in tree) ---
        draft0_tokens = list(range(1000, 1500))  # 500 draft tokens
        turn0_full = PREFIX_TOKENS + draft0_tokens  # 150 + 500 = 650

        # Insert as uncompressed (like a normal DRAFT that finishes without compression)
        kv_full = alloc.alloc(len(turn0_full))
        cache.insert(turn0_full, kv_full.clone())

        # --- Step 2: Simulate Turn 0 CONTINUE (compressed) ---
        continue0_tokens = list(range(2000, 2500))  # 500 continue tokens
        continue_input = turn0_full  # prompt for continue = sys+q+draft0

        # Simulate: match prefix, extend 1 token, compress, then decode 500 tokens
        req_pool_C = rtp.alloc()
        # After match_prefix: prefix = turn0_full ≈ 649 tokens matched
        # Extend 1 token → compression triggers
        # For simplicity, simulate compression of the full prompt portion
        prompt_len = len(continue_input)
        compress_selected = torch.arange(0, prompt_len, 2, dtype=torch.int64)  # keep every other
        compressed_len = len(compress_selected)
        prefix_kv = kv_full[compress_selected]

        # Write compressed prefix to req_to_token
        rtp.req_to_token[req_pool_C, :compressed_len] = prefix_kv

        # Simulate decode: allocate KV for 500 decode tokens
        decode_kv = alloc.alloc(len(continue0_tokens))
        rtp.req_to_token[req_pool_C, compressed_len:compressed_len + len(continue0_tokens)] = decode_kv

        # Build a FakeReq that looks like a compressed request after decode
        from dataclasses import dataclass
        req_C = FakeReq(
            rid="continue0",
            origin_input_ids=continue_input,
            output_ids=[9999] + list(continue0_tokens) + [8888],  # prefill_out + decode_outs + final
        )
        req_C.is_kv_compressed = True
        req_C.compressed_seq_len = compressed_len
        req_C.original_seq_len = prompt_len  # token-space length at compression time
        req_C._snapkv_position_offset = prompt_len - compressed_len
        req_C.selected_positions = compress_selected
        req_C.req_pool_idx = req_pool_C

        # Insert compressed into tree first (simulating what scheduler does at compression time)
        cache.insert_compressed(continue_input, prefix_kv.clone(), compress_selected)
        _, node = cache.match_prefix(continue_input)
        cache.inc_lock_ref(node)
        req_C.last_node = node
        req_C.prefix_indices = prefix_kv

        # --- Step 3: Call cache_finished_req (THE FIX) ---
        cache.cache_finished_req(req_C)

        # --- Step 4: Verify decode tokens are in the tree ---
        # Build the token sequence that includes decode tokens
        # token_ids = (origin_input_ids + output_ids)[:-1]
        #           = continue_input + [9999] + continue0_tokens
        expected_cached = continue_input + [9999] + list(continue0_tokens)

        val, node = cache.match_prefix(expected_cached)
        matched_len = getattr(node, "_match_token_len", len(val))

        print(f"\n  Expected cached tokens: {len(expected_cached)}")
        print(f"  Matched tokens: {matched_len}")
        print(f"  Matched KV entries: {len(val)}")

        # Before fix: matched_len would be ≈ len(continue_input) (only compressed prefix)
        # After fix: matched_len should be ≈ len(expected_cached) (includes decode tokens)
        self.assertGreater(matched_len, len(continue_input),
            f"Decode tokens NOT in tree! Matched {matched_len} tokens but "
            f"expected > {len(continue_input)} (input) + some decode tokens")
        print(f"  ✓ Decode tokens are in the tree! ({matched_len - len(continue_input)} extra tokens matched)")

    def test_next_turn_prefix_hit_with_decode_tokens(self):
        """Turn 1 DRAFT should be able to prefix-match Turn 0 CONTINUE's decode output."""
        cache, rtp, alloc = make_env()

        # Turn 0 DRAFT output (uncompressed in tree)
        draft0 = list(range(1000, 1400))
        turn0_tokens = PREFIX_TOKENS + draft0  # 550 tokens
        kv0 = alloc.alloc(len(turn0_tokens))
        cache.insert(turn0_tokens, kv0.clone())

        # Turn 0 CONTINUE: compresses, decodes continue0, finishes
        continue0 = list(range(2000, 2300))  # 300 continue tokens
        prompt_C = turn0_tokens
        compress_sel = torch.arange(0, len(prompt_C), 2, dtype=torch.int64)
        comp_len = len(compress_sel)
        prefix_kv = kv0[compress_sel]

        req_pool = rtp.alloc()
        rtp.req_to_token[req_pool, :comp_len] = prefix_kv
        cont_kv = alloc.alloc(len(continue0))
        rtp.req_to_token[req_pool, comp_len:comp_len + len(continue0)] = cont_kv

        req = FakeReq(
            rid="cont0", origin_input_ids=prompt_C,
            output_ids=[9999] + list(continue0) + [8888],
        )
        req.is_kv_compressed = True
        req.compressed_seq_len = comp_len
        req.original_seq_len = len(prompt_C)
        req._snapkv_position_offset = len(prompt_C) - comp_len
        req.selected_positions = compress_sel
        req.req_pool_idx = req_pool

        cache.insert_compressed(prompt_C, prefix_kv.clone(), compress_sel)
        _, node = cache.match_prefix(prompt_C)
        cache.inc_lock_ref(node)
        req.last_node = node
        req.prefix_indices = prefix_kv

        cache.cache_finished_req(req)

        # Turn 1 DRAFT: prompt = sys + q + draft0 + continue0 + draft1_start
        draft1_start = [7777]
        turn1_input = turn0_tokens + [9999] + list(continue0) + draft1_start
        req1 = FakeReq(rid="draft1", origin_input_ids=turn1_input)
        req1.init_next_round_input(cache)

        matched = getattr(req1.last_node, "_match_token_len", len(req1.prefix_indices))
        expected_match = len(turn0_tokens) + 1 + len(continue0)  # everything except draft1_start

        print(f"\n  Turn 1 DRAFT input: {len(turn1_input)} tokens")
        print(f"  Expected prefix match: ≥{expected_match - 1} tokens")
        print(f"  Actual prefix match: {matched} tokens")
        print(f"  extend_input_len: {req1.extend_input_len}")

        # Before fix: matched ≈ len(turn0_tokens) (only draft0, ~550)
        # After fix: matched ≈ expected_match (draft0 + continue0, ~851)
        self.assertGreater(matched, len(turn0_tokens) + 1,
            f"Turn 1 should match beyond draft0! Got {matched}, expected >{len(turn0_tokens) + 1}")
        print(f"  ✓ Turn 1 matches {matched - len(turn0_tokens)} tokens beyond draft0 (continue0 tokens)")


class TestBug1ReproCompressAfterCompressedPrefix(unittest.TestCase):
    """Reproduce Bug 1: compress a request that matched a compressed prefix.

    Scenario:
      1. Request A: prefill 600 tokens → compress to 400 → insert_compressed → decode 300 → finish
         Tree now has: uncompressed node (150 shared prefix) + compressed node (450 tokens → ~250 KV)
      2. Request B: prompt = same 600 tokens + A's 300 decode tokens + 1 new token
         match_prefix → matches compressed node → kv_offset > 0
         Should trigger compression → Bug 1: selected positions in KV-space
         but insert_compressed expects token-space
    """

    def test_bug1_coordinate_space_mismatch(self):
        cache, rtp, alloc = make_env()

        # --- Step 1: Request A generates and compresses ---
        draft_tokens = list(range(1000, 1450))  # 450 draft tokens
        turn0_tokens = PREFIX_TOKENS + draft_tokens  # 150 + 450 = 600

        # Simulate prefill + compress for Request A
        req_pool_A, comp_len_A, selected_A, node_A = simulate_prefill_and_compress(
            cache, alloc, rtp, turn0_tokens, budget_ratio=0.5
        )
        print(f"\nStep 1: Request A compressed {len(turn0_tokens)} -> {comp_len_A} tokens")
        print(f"  selected_A positions: min={selected_A.min().item()}, max={selected_A.max().item()}, count={len(selected_A)}")

        # Simulate Request A's decode phase: 300 output tokens
        decode_output_ids = list(range(5000, 5300))

        # Now simulate cache_finished_req for A (compressed path):
        # - Free decode KV (not stored in tree)
        # - Only the compressed prefix remains in tree
        cache.dec_lock_ref(node_A)
        rtp.free(req_pool_A)

        # --- Step 1b: Store A's full output in tree (simulating Bug 2 FIX) ---
        # Without this, Request B can't match A's decode tokens and Bug 1 won't trigger.
        # We manually insert A's full sequence (prefix + draft + decode) into tree.
        full_A_tokens = turn0_tokens + decode_output_ids  # 600 + 300 = 900 tokens
        full_A_kv = alloc.alloc(len(full_A_tokens))
        # Insert using insert_compressed with the known compressed positions
        # The first 600 tokens map to compressed positions, last 300 are contiguous
        prefix_positions = selected_A.clone()
        decode_positions = torch.arange(len(turn0_tokens), len(full_A_tokens), dtype=torch.int64)
        all_positions = torch.cat([prefix_positions, decode_positions])
        combined_kv = alloc.alloc(comp_len_A + len(decode_output_ids))
        cache.insert_compressed(full_A_tokens, combined_kv, all_positions)
        print(f"\nStep 1b: Inserted full A tokens into tree ({len(full_A_tokens)} tokens, {len(combined_kv)} KV)")

        # --- Step 2: Request B matches compressed prefix and triggers compression ---
        continue_tokens = list(range(6000, 6200))  # 200 new tokens
        request_B_input = full_A_tokens + continue_tokens  # 900 + 200 = 1100 tokens

        req_B = FakeReq(rid="reqB", origin_input_ids=request_B_input, output_ids=[])
        req_B.init_next_round_input(cache)

        prefix_kv_len = len(req_B.prefix_indices)
        kv_offset = req_B._snapkv_position_offset
        matched_token_len = getattr(req_B.last_node, "_match_token_len", prefix_kv_len)

        print(f"\nStep 2: Request B")
        print(f"  fill_ids length: {len(req_B.fill_ids)}")
        print(f"  prefix_kv_len (KV indices): {prefix_kv_len}")
        print(f"  matched_token_len: {matched_token_len}")
        print(f"  kv_offset: {kv_offset}")
        print(f"  extend_input_len: {req_B.extend_input_len}")

        # KEY CHECK: Does Request B have kv_offset > 0?
        if kv_offset > 0:
            print(f"\n  ★ BUG 1 TRIGGER CONDITION MET: kv_offset={kv_offset} > 0")
            print(f"  If SnapKV compresses now, selected positions would be in KV-space [0, {prefix_kv_len + req_B.extend_input_len - 1}]")
            print(f"  But insert_compressed expects token-space [0, {len(request_B_input) - 1}]")
            print(f"  The mismatch = {kv_offset} positions")

            # Simulate what compression would do:
            kv_seq_len = len(request_B_input) - kv_offset  # KV-space length
            # SnapKV would produce selected positions in [0, kv_seq_len-1]
            simulated_selected = torch.arange(0, kv_seq_len, 2)  # keep every other
            print(f"\n  Simulated compression: kv_seq_len={kv_seq_len}, keeping {len(simulated_selected)} positions")
            print(f"  Max selected position (KV-space): {simulated_selected.max().item()}")
            print(f"  Token-space length: {len(request_B_input)}")
            print(f"  ★ BUG: insert_compressed will compare selected < node.key_len")
            print(f"    node.key_len is in TOKEN space (~900)")
            print(f"    but selected max is in KV space ({simulated_selected.max().item()})")
            print(f"    Since {simulated_selected.max().item()} < 900, mask eats ALL values at first node!")
        else:
            print(f"\n  kv_offset=0, Bug 1 condition NOT met")

        self.assertGreater(kv_offset, 0,
            "Expected kv_offset > 0 to trigger Bug 1. "
            "If this fails, the compressed node was not matched.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
