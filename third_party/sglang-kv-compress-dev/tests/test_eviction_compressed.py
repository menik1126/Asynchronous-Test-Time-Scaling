"""Unit tests for high-memory-pressure eviction with compressed KV nodes.

Reproduces the bug where re-compression of a request that matched a
compressed prefix reads stale data from req_to_token because req.seqlen
(token space) > actual valid KV entries (KV space).

Root cause:
  scheduler.py uses ``req.seqlen`` (token-space length) to read
  ``req_to_token[:seqlen]`` for SnapKV compression.  But when a
  request matched a compressed prefix, the valid KV entries in
  req_to_token only go up to ``kv_pre_len + extend_input_len`` (the
  KV-space length), which is less than ``req.seqlen`` by
  ``position_offset``.  The stale entries beyond the valid range are
  treated as "unselected" KV and freed — corrupting the pool and
  causing CUDA index-out-of-bounds on later eviction.

Run:
  python tests/test_eviction_compressed.py
"""
from __future__ import annotations

import unittest
from typing import Set

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode


# ---------------------------------------------------------------------------
# Stubs with slot tracking
# ---------------------------------------------------------------------------

class TrackingReqToTokenPool:
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


class TrackingAllocator:
    """Tracks allocated/freed slots to detect double-frees and invalid frees."""

    def __init__(self):
        self._next_slot = 100  # Start at 100 to make stale 0s detectable
        self.allocated: Set[int] = set()
        self.freed_history = []
        self.double_frees = []
        self.invalid_frees = []

    def alloc(self, n):
        slots = list(range(self._next_slot, self._next_slot + n))
        self._next_slot += n
        self.allocated.update(slots)
        return torch.tensor(slots, dtype=torch.int32)

    def free(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        for idx in indices:
            if idx not in self.allocated:
                self.invalid_frees.append(idx)
            elif idx not in self.allocated:
                self.double_frees.append(idx)
            else:
                self.allocated.discard(idx)
            self.freed_history.append(idx)

    def available_size(self):
        return 99999


def make_env():
    rtp = TrackingReqToTokenPool()
    alloc = TrackingAllocator()
    cache = RadixCache(rtp, alloc, disable=False)
    return cache, rtp, alloc


# ===================================================================
# Bug 1: Stale data in req_to_token when re-compressing
# ===================================================================

class TestStaleDataReCompression(unittest.TestCase):
    """Reproduces the stale-data bug in the scheduler's compression path.

    Scenario:
      1. Request A: prefill(sys+Q) → compress → insert_compressed
      2. Request B: match compressed prefix → extend with CoT →
         re-compress using req.seqlen (token space)
      3. req_to_token[:seqlen] reads beyond valid KV → stale indices freed
    """

    def test_stale_entries_beyond_kv_seq_len(self):
        """Demonstrate that req_to_token[:token_seqlen] reads stale data."""
        cache, rtp, alloc = make_env()

        # === Step 1: First request compresses sys+Q ===
        token_ids_a = list(range(1, 151))  # 150 tokens
        kv_a = alloc.alloc(150)
        req_pool_a = rtp.alloc()
        rtp.req_to_token[req_pool_a, :150] = kv_a

        # Compress: keep every other position → 75 KV entries
        selected_a = torch.tensor(list(range(0, 150, 2)), dtype=torch.int64)
        selected_kv_a = kv_a[selected_a]

        # Free unselected
        unselected_a = torch.tensor(list(range(1, 150, 2)), dtype=torch.int64)
        alloc.free(kv_a[unselected_a])

        # Write compressed to req_to_token
        rtp.req_to_token[req_pool_a, :75] = selected_kv_a

        # Insert to tree
        cache.insert_compressed(token_ids_a, selected_kv_a.clone(), selected_a)

        # === Step 2: Request B matches compressed prefix and extends ===
        req_pool_b = rtp.alloc()

        # Write STALE data into req_to_token row (simulating leftover from previous use)
        rtp.req_to_token[req_pool_b, :] = 9999  # Stale sentinel value

        # Match compressed prefix → gets 75 KV entries
        prefix_kv, node = cache.match_prefix(token_ids_a)
        kv_pre_len = len(prefix_kv)  # 75 (compressed)
        matched_token_len = node._match_token_len  # 150 (original)
        position_offset = node._match_position_offset  # 75

        # Write compressed prefix to req_to_token
        rtp.req_to_token[req_pool_b, :kv_pre_len] = prefix_kv

        # Extend with 50 new tokens
        extend_kv = alloc.alloc(50)
        rtp.req_to_token[req_pool_b, kv_pre_len:kv_pre_len + 50] = extend_kv

        # Valid KV range: [0, 125) = 75 prefix + 50 extend
        kv_seq_len = kv_pre_len + 50  # 125

        # Token-space seq_len (what req.seqlen would return)
        token_seq_len = matched_token_len + 50 + 1  # 150 + 50 + 1 output = 201

        # === Step 3: Read req_to_token with WRONG (token) seq_len ===
        full_kv_wrong = rtp.req_to_token[req_pool_b, :token_seq_len].clone()

        # The entries at [125:201] are STALE (9999)
        stale_entries = full_kv_wrong[kv_seq_len:]
        self.assertTrue(
            (stale_entries == 9999).all(),
            f"Expected stale data beyond KV seq_len, got {stale_entries[:5]}"
        )

        # If SnapKV "unselects" some stale entries and frees them, pool corrupts
        alloc.invalid_frees.clear()
        alloc.free(torch.tensor([9999]))  # Simulates freeing a stale index
        self.assertEqual(len(alloc.invalid_frees), 1,
                         "Freeing stale index 9999 should be detected as invalid")

        # === Step 3b: Read with CORRECT (KV) seq_len ===
        full_kv_correct = rtp.req_to_token[req_pool_b, :kv_seq_len].clone()

        # No stale entries
        valid_prefix = full_kv_correct[:kv_pre_len]
        valid_extend = full_kv_correct[kv_pre_len:]
        self.assertTrue(
            (valid_prefix == prefix_kv).all(),
            "Prefix portion should match tree's compressed KV"
        )
        self.assertTrue(
            (valid_extend == extend_kv).all(),
            "Extend portion should match allocated KV"
        )

    def test_correct_kv_seq_len_formula(self):
        """Verify the correct formula for KV-space seq_len."""
        # For a request that matched a compressed prefix:
        #   kv_pre_len = len(prefix_indices) = compressed KV count
        #   extend_input_len = len(fill_ids) - matched_token_len
        #   kv_seq_len = kv_pre_len + extend_input_len
        #   token_seq_len = len(origin_input_ids) + len(output_ids)
        #   position_offset = matched_token_len - kv_pre_len
        #
        # Relationship:
        #   kv_seq_len = token_seq_len - position_offset - len(output_ids)
        #   (approximately, depends on when output_ids is set)

        kv_pre_len = 75
        matched_token_len = 150
        position_offset = 75  # matched_token_len - kv_pre_len
        extend_input_len = 50

        kv_seq_len = kv_pre_len + extend_input_len  # 125
        token_seq_len_approx = matched_token_len + extend_input_len + 1  # 201

        self.assertEqual(kv_seq_len, 125)
        self.assertEqual(token_seq_len_approx, 201)
        self.assertGreater(token_seq_len_approx, kv_seq_len,
                           "Token seq_len should be larger than KV seq_len for compressed requests")
        self.assertEqual(token_seq_len_approx - kv_seq_len, position_offset + 1,
                         "Difference should be position_offset + 1 (output token)")


# ===================================================================
# Bug 2: Eviction after pool corruption
# ===================================================================

class TestEvictionAfterPoolCorruption(unittest.TestCase):
    """Simulates the crash scenario: evict compressed node after
    pool was corrupted by freeing stale indices.
    """

    def test_eviction_of_valid_compressed_node(self):
        """A properly created compressed node should evict cleanly."""
        cache, rtp, alloc = make_env()

        kv = alloc.alloc(5)
        cache.insert_compressed(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            kv,
            torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64),
        )

        alloc.invalid_frees.clear()
        cache.evict(100, alloc.free)

        self.assertEqual(len(alloc.invalid_frees), 0,
                         f"Evicting valid compressed node should not trigger invalid frees: {alloc.invalid_frees}")

    def test_eviction_after_stale_free_corrupts_pool(self):
        """After freeing stale indices, subsequent eviction fails."""
        cache, rtp, alloc = make_env()

        # Insert compressed node
        kv = alloc.alloc(5)
        cache.insert_compressed(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            kv,
            torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64),
        )

        # Simulate pool corruption: free a slot that's owned by the tree
        slot_owned_by_tree = kv[2].item()
        alloc.free(torch.tensor([slot_owned_by_tree]))  # BUG: double-free path

        # Now evict the tree node
        alloc.invalid_frees.clear()
        alloc.double_frees.clear()
        cache.evict(100, alloc.free)

        # The evicted node tries to free slot_owned_by_tree again
        freed_slots = alloc.freed_history[-5:]
        self.assertIn(slot_owned_by_tree, freed_slots,
                      "Eviction should try to free the tree's slots")


# ===================================================================
# Bug 3: Re-compression with correct KV seq_len should be safe
# ===================================================================

class TestReCompressionSafe(unittest.TestCase):
    """Verify that using kv_seq_len (not token_seq_len) for
    re-compression avoids stale data and invalid frees.
    """

    def test_no_invalid_frees_with_correct_seq_len(self):
        """Full simulation: compress → match → extend → re-compress with correct seq_len."""
        cache, rtp, alloc = make_env()

        # Step 1: First compression
        tokens_a = list(range(1, 101))  # 100 tokens
        kv_a = alloc.alloc(100)
        req_pool = rtp.alloc()
        rtp.req_to_token[req_pool, :100] = kv_a

        selected_a = torch.tensor(list(range(0, 100, 2)), dtype=torch.int64)  # 50 entries
        selected_kv_a = kv_a[selected_a]

        # Free unselected
        unselected_a = torch.tensor(list(range(1, 100, 2)), dtype=torch.int64)
        alloc.free(kv_a[unselected_a])

        rtp.req_to_token[req_pool, :50] = selected_kv_a
        cache.insert_compressed(tokens_a, selected_kv_a.clone(), selected_a)

        # Step 2: New request matches and extends
        req_pool2 = rtp.alloc()
        rtp.req_to_token[req_pool2, :] = -1  # Fill with invalid sentinel

        prefix_kv, node = cache.match_prefix(tokens_a)
        kv_pre_len = len(prefix_kv)  # 50
        rtp.req_to_token[req_pool2, :kv_pre_len] = prefix_kv

        extend_kv = alloc.alloc(30)
        rtp.req_to_token[req_pool2, kv_pre_len:kv_pre_len + 30] = extend_kv

        kv_seq_len = kv_pre_len + 30  # 80 (correct)
        token_seq_len = 100 + 30 + 1  # 131 (WRONG for req_to_token access)

        # Step 3: Re-compress using CORRECT kv_seq_len
        alloc.invalid_frees.clear()
        full_kv = rtp.req_to_token[req_pool2, :kv_seq_len].clone()

        # Simulate SnapKV: keep every other entry
        new_selected = torch.tensor(list(range(0, kv_seq_len, 2)), dtype=torch.int64)
        keep_mask = torch.zeros(kv_seq_len, dtype=torch.bool)
        keep_mask[new_selected] = True

        # Protect prefix
        unsel_mask = ~keep_mask
        unsel_mask[:kv_pre_len] = False

        unsel_pos = unsel_mask.nonzero(as_tuple=True)[0]
        if unsel_pos.numel() > 0:
            alloc.free(full_kv[unsel_pos])

        # No invalid frees should happen
        self.assertEqual(len(alloc.invalid_frees), 0,
                         f"Using kv_seq_len should not cause invalid frees: {alloc.invalid_frees}")

    def test_invalid_frees_with_wrong_seq_len(self):
        """Same scenario but using token_seq_len — should detect invalid frees."""
        cache, rtp, alloc = make_env()

        tokens_a = list(range(1, 101))
        kv_a = alloc.alloc(100)
        req_pool = rtp.alloc()
        rtp.req_to_token[req_pool, :100] = kv_a

        selected_a = torch.tensor(list(range(0, 100, 2)), dtype=torch.int64)
        selected_kv_a = kv_a[selected_a]
        unselected_a = torch.tensor(list(range(1, 100, 2)), dtype=torch.int64)
        alloc.free(kv_a[unselected_a])

        rtp.req_to_token[req_pool, :50] = selected_kv_a
        cache.insert_compressed(tokens_a, selected_kv_a.clone(), selected_a)

        req_pool2 = rtp.alloc()
        rtp.req_to_token[req_pool2, :] = 9999  # Stale sentinel

        prefix_kv, node = cache.match_prefix(tokens_a)
        kv_pre_len = len(prefix_kv)
        rtp.req_to_token[req_pool2, :kv_pre_len] = prefix_kv

        extend_kv = alloc.alloc(30)
        rtp.req_to_token[req_pool2, kv_pre_len:kv_pre_len + 30] = extend_kv

        kv_seq_len = kv_pre_len + 30
        token_seq_len = 100 + 30 + 1  # WRONG

        # Re-compress using WRONG token_seq_len
        alloc.invalid_frees.clear()
        full_kv_wrong = rtp.req_to_token[req_pool2, :token_seq_len].clone()

        new_selected = torch.tensor(list(range(0, token_seq_len, 2)), dtype=torch.int64)
        keep_mask = torch.zeros(token_seq_len, dtype=torch.bool)
        keep_mask[new_selected] = True
        unsel_mask = ~keep_mask
        unsel_mask[:kv_pre_len] = False

        unsel_pos = unsel_mask.nonzero(as_tuple=True)[0]
        if unsel_pos.numel() > 0:
            alloc.free(full_kv_wrong[unsel_pos])

        # Should detect invalid frees (stale 9999 values)
        self.assertGreater(len(alloc.invalid_frees), 0,
                           "Using token_seq_len should cause invalid frees from stale data")
        self.assertIn(9999, alloc.invalid_frees,
                      "Stale sentinel value 9999 should appear in invalid frees")


# ===================================================================
# Bug 4: Compressed node eviction with lock_ref lifecycle
# ===================================================================

class TestCompressedEvictionLifecycle(unittest.TestCase):

    def test_locked_node_not_evicted(self):
        """Compressed node with lock_ref > 0 must not be evicted."""
        cache, rtp, alloc = make_env()

        kv = alloc.alloc(5)
        cache.insert_compressed(
            [1, 2, 3, 4, 5],
            kv,
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
        )
        _, node = cache.match_prefix([1, 2, 3, 4, 5])
        cache.inc_lock_ref(node)

        freed_before = len(alloc.freed_history)
        cache.evict(100, alloc.free)
        freed_after = len(alloc.freed_history)

        self.assertEqual(freed_after, freed_before,
                         "Locked compressed node should not be evicted")

    def test_unlocked_node_evicted_cleanly(self):
        """After unlock, compressed node should be evictable."""
        cache, rtp, alloc = make_env()

        kv = alloc.alloc(5)
        slot_values = kv.tolist()
        cache.insert_compressed(
            [1, 2, 3, 4, 5],
            kv,
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
        )
        _, node = cache.match_prefix([1, 2, 3, 4, 5])
        cache.inc_lock_ref(node)
        cache.dec_lock_ref(node)

        alloc.invalid_frees.clear()
        cache.evict(100, alloc.free)

        self.assertEqual(len(alloc.invalid_frees), 0,
                         "Evicting unlocked compressed node should not cause invalid frees")
        for s in slot_values:
            self.assertNotIn(s, alloc.allocated,
                             f"Slot {s} should have been freed by eviction")


if __name__ == "__main__":
    unittest.main(verbosity=2)
