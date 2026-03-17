"""Unit tests for compressed KV prefix tree (RadixCache).

Tests insert_compressed, match_prefix with compressed nodes,
_split_node on compressed nodes, eviction, and lock_ref accounting.

Run:
  python tests/test_radix_cache_compressed.py
"""
from __future__ import annotations

import sys
import unittest

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode


# ---------------------------------------------------------------------------
# Lightweight stubs so we don't need a full SGLang runtime
# ---------------------------------------------------------------------------

class FakeReqToTokenPool:
    def __init__(self, max_reqs=8, max_len=2048):
        self.req_to_token = torch.zeros((max_reqs, max_len), dtype=torch.int32)

    def free(self, idx):
        pass

    def write(self, loc, val):
        req_idx, slc = loc
        self.req_to_token[req_idx, slc] = val


class FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        if isinstance(indices, torch.Tensor):
            self.freed.extend(indices.tolist())
        else:
            self.freed.extend(list(indices))


def make_cache(disable=False):
    return RadixCache(FakeReqToTokenPool(), FakeAllocator(), disable=disable)


def get_alloc(cache):
    return cache.token_to_kv_pool_allocator


# ===================================================================
# 1. Basic insert_compressed + match_prefix
# ===================================================================

class TestInsertCompressedBasic(unittest.TestCase):

    def test_insert_and_full_match(self):
        cache = make_cache()
        key = [10, 20, 30, 40, 50]
        kv = torch.tensor([100, 300, 500], dtype=torch.int32)
        sel = torch.tensor([0, 2, 4], dtype=torch.int64)

        cache.insert_compressed(key, kv, sel)
        value, node = cache.match_prefix(key)

        self.assertEqual(value.tolist(), [100, 300, 500])
        self.assertEqual(node._match_token_len, 5)
        self.assertEqual(node._match_position_offset, 2)

    def test_match_longer_key(self):
        cache = make_cache()
        key = [10, 20, 30, 40, 50]
        kv = torch.tensor([100, 300, 500], dtype=torch.int32)
        sel = torch.tensor([0, 2, 4], dtype=torch.int64)
        cache.insert_compressed(key, kv, sel)

        value, node = cache.match_prefix([10, 20, 30, 40, 50, 60, 70])
        self.assertEqual(value.tolist(), [100, 300, 500])
        self.assertEqual(node._match_token_len, 5)
        self.assertEqual(node._match_position_offset, 2)

    def test_match_shorter_key_triggers_split(self):
        cache = make_cache()
        key = [10, 20, 30, 40, 50]
        kv = torch.tensor([100, 300, 500], dtype=torch.int32)
        sel = torch.tensor([0, 2, 4], dtype=torch.int64)
        cache.insert_compressed(key, kv, sel)

        value, node = cache.match_prefix([10, 20, 30])
        # Positions 0,2 fall in key[:3]; position 4 does not
        self.assertEqual(value.tolist(), [100, 300])
        self.assertEqual(node._match_token_len, 3)
        self.assertEqual(node._match_position_offset, 1)

    def test_no_match(self):
        cache = make_cache()
        cache.insert_compressed(
            [10, 20, 30],
            torch.tensor([100, 200], dtype=torch.int32),
            torch.tensor([0, 2], dtype=torch.int64),
        )
        value, node = cache.match_prefix([99, 88])
        self.assertEqual(len(value), 0)
        self.assertEqual(node._match_token_len, 0)
        self.assertEqual(node._match_position_offset, 0)


# ===================================================================
# 2. _split_node on compressed nodes
# ===================================================================

class TestSplitCompressed(unittest.TestCase):

    def test_split_preserves_total_kv(self):
        cache = make_cache()
        key = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        kv = torch.tensor([10, 40, 60, 90], dtype=torch.int32)
        sel = torch.tensor([0, 3, 5, 8], dtype=torch.int64)
        cache.insert_compressed(key, kv, sel)

        # Match first 6 tokens → split at key position 6
        value, node = cache.match_prefix([1, 2, 3, 4, 5, 6])
        # Positions < 6: [0, 3, 5] → 3 kv entries
        self.assertEqual(value.tolist(), [10, 40, 60])
        self.assertEqual(node._match_token_len, 6)
        self.assertEqual(node._match_position_offset, 3)

        # Full key should still return all 4 KV entries
        value_full, node_full = cache.match_prefix(key)
        self.assertEqual(value_full.tolist(), [10, 40, 60, 90])
        self.assertEqual(node_full._match_token_len, 10)
        self.assertEqual(node_full._match_position_offset, 6)

    def test_split_no_kv_in_first_half(self):
        cache = make_cache()
        key = [1, 2, 3, 4, 5]
        kv = torch.tensor([300, 400], dtype=torch.int32)
        sel = torch.tensor([3, 4], dtype=torch.int64)
        cache.insert_compressed(key, kv, sel)

        value, node = cache.match_prefix([1, 2])
        self.assertEqual(value.tolist(), [])
        self.assertEqual(node._match_token_len, 2)
        self.assertEqual(node._match_position_offset, 2)


# ===================================================================
# 3. Mixed compressed + uncompressed nodes
# ===================================================================

class TestMixed(unittest.TestCase):

    def test_uncompressed_then_compressed(self):
        cache = make_cache()
        # Uncompressed [1,2,3]
        cache.insert([1, 2, 3], torch.tensor([10, 20, 30], dtype=torch.int32))

        # Compressed [1,2,3,4,5,6] — tree already has [1,2,3],
        # so the new compressed node covers [4,5,6] keeping positions 3,5
        full_key = [1, 2, 3, 4, 5, 6]
        compressed_kv = torch.tensor([10, 20, 30, 40, 60], dtype=torch.int32)
        sel = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int64)
        cache.insert_compressed(full_key, compressed_kv, sel)

        value, node = cache.match_prefix(full_key)
        self.assertEqual(node._match_token_len, 6)
        self.assertEqual(node._match_position_offset, 1)
        self.assertEqual(len(value), 5)

    def test_compressed_then_uncompressed_extend(self):
        cache = make_cache()
        # Compressed [1,2,3,4,5] → keep 0,2,4
        cache.insert_compressed(
            [1, 2, 3, 4, 5],
            torch.tensor([10, 30, 50], dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int64),
        )
        # Uncompressed extend [1,2,3,4,5,6,7,8]
        cache.insert(
            [1, 2, 3, 4, 5, 6, 7, 8],
            torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32),
        )

        value, node = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(node._match_token_len, 8)
        self.assertEqual(node._match_position_offset, 2)
        self.assertEqual(len(value), 6)  # 3 compressed + 3 uncompressed


# ===================================================================
# 4. evictable_size tracking
# ===================================================================

class TestEvictableSize(unittest.TestCase):

    def test_compressed_evictable_size(self):
        cache = make_cache()
        cache.insert_compressed(
            [1, 2, 3, 4, 5],
            torch.tensor([10, 30, 50], dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int64),
        )
        self.assertEqual(cache.evictable_size(), 3)

    def test_eviction_frees_compressed_value(self):
        cache = make_cache()
        alloc = get_alloc(cache)
        cache.insert_compressed(
            [1, 2, 3, 4, 5],
            torch.tensor([10, 30, 50], dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int64),
        )
        cache.evict(100, alloc.free)
        self.assertEqual(sorted(alloc.freed), [10, 30, 50])
        self.assertEqual(cache.evictable_size(), 0)

    def test_total_size_compressed(self):
        cache = make_cache()
        cache.insert_compressed(
            [1, 2, 3, 4, 5],
            torch.tensor([10, 30, 50], dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int64),
        )
        self.assertEqual(cache.total_size(), 3)


# ===================================================================
# 5. lock_ref accounting with compressed nodes
# ===================================================================

class TestLockRef(unittest.TestCase):

    def test_lock_unlock_uses_value_len(self):
        cache = make_cache()
        cache.insert_compressed(
            [1, 2, 3, 4, 5],
            torch.tensor([10, 30, 50], dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int64),
        )
        _, node = cache.match_prefix([1, 2, 3, 4, 5])

        self.assertEqual(cache.evictable_size(), 3)
        self.assertEqual(cache.protected_size(), 0)

        cache.inc_lock_ref(node)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 3)

        cache.dec_lock_ref(node)
        self.assertEqual(cache.evictable_size(), 3)
        self.assertEqual(cache.protected_size(), 0)


# ===================================================================
# 6. Multiple requests reusing the same compressed prefix
# ===================================================================

class TestPrefixReuse(unittest.TestCase):

    def test_two_requests_same_prefix(self):
        cache = make_cache()
        prefix = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        kv = torch.tensor([10, 30, 50, 80, 100], dtype=torch.int32)
        sel = torch.tensor([0, 2, 4, 7, 9], dtype=torch.int64)
        cache.insert_compressed(prefix, kv, sel)

        val_a, node_a = cache.match_prefix(prefix)
        self.assertEqual(val_a.tolist(), [10, 30, 50, 80, 100])
        self.assertEqual(node_a._match_position_offset, 5)

        val_b, node_b = cache.match_prefix(prefix + [11, 12, 13])
        self.assertEqual(val_b.tolist(), [10, 30, 50, 80, 100])
        self.assertEqual(node_b._match_token_len, 10)

    def test_different_extensions(self):
        cache = make_cache()
        prefix = [1, 2, 3, 4, 5]
        kv = torch.tensor([10, 30, 50], dtype=torch.int32)
        sel = torch.tensor([0, 2, 4], dtype=torch.int64)
        cache.insert_compressed(prefix, kv, sel)

        val_a, _ = cache.match_prefix([1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(val_a), 3)

        val_b, _ = cache.match_prefix([1, 2, 3, 4, 5, 8, 9])
        self.assertEqual(len(val_b), 3)


# ===================================================================
# 7. Backward compatibility — uncompressed operations
# ===================================================================

class TestBackwardCompat(unittest.TestCase):

    def test_normal_insert_match(self):
        cache = make_cache()
        key = [1, 2, 3, 4, 5]
        val = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        cache.insert(key, val)

        result, node = cache.match_prefix(key)
        self.assertEqual(result.tolist(), [10, 20, 30, 40, 50])
        self.assertEqual(node._match_token_len, 5)
        self.assertEqual(node._match_position_offset, 0)

    def test_normal_split(self):
        cache = make_cache()
        cache.insert([1, 2, 3, 4, 5], torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32))

        val, node = cache.match_prefix([1, 2, 3])
        self.assertEqual(val.tolist(), [10, 20, 30])
        self.assertEqual(node._match_token_len, 3)
        self.assertEqual(node._match_position_offset, 0)

    def test_normal_eviction(self):
        cache = make_cache()
        alloc = get_alloc(cache)
        cache.insert([1, 2, 3], torch.tensor([10, 20, 30], dtype=torch.int32))
        self.assertEqual(cache.evictable_size(), 3)

        cache.evict(100, alloc.free)
        self.assertEqual(sorted(alloc.freed), [10, 20, 30])
        self.assertEqual(cache.evictable_size(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
