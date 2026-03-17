from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()

        self.hit_count = 0
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

        # KV compression: when position_offset > 0, len(value) < len(key).
        # selected_positions maps each value entry to its original key position.
        self.position_offset = 0
        self.selected_positions = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        disable: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.disable = disable
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: List[int], **kwargs) -> Tuple[torch.Tensor, "TreeNode"]:
        """Find the matching prefix from the radix tree.

        Returns:
            (kv_indices, last_node) where kv_indices may be shorter than
            the number of matched tokens when compressed nodes are on the
            path.  Callers should use ``last_node`` to retrieve
            ``position_offset`` accumulated along the matched path via
            :meth:`get_match_position_offset`.
        """
        if self.disable:
            return [], self.root_node

        value, last_node, matched_key_len, total_offset, all_selected = (
            self._match_prefix_helper(self.root_node, key)
        )
        if value:
            value = torch.concat(value)
        else:
            value = torch.tensor([], dtype=torch.int32)
        last_node._match_position_offset = total_offset
        last_node._match_token_len = matched_key_len
        last_node._match_selected_positions = all_selected
        return value, last_node

    def insert(self, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def insert_compressed(
        self,
        key: List,
        compressed_value: torch.Tensor,
        selected_positions: torch.Tensor,
    ):
        """Insert compressed KV into the tree.

        ``key`` is the full token-ID sequence (for prefix matching).
        ``compressed_value`` holds only the selected KV-slot indices
        (``len(compressed_value) <= len(key)``).
        ``selected_positions`` maps each value entry to its position in
        *key*-space (sorted, same length as ``compressed_value``).
        """
        if self.disable:
            return

        node = self.root_node
        node.last_access_time = time.time()
        remaining_key = list(key)
        remaining_value = compressed_value
        remaining_sel = selected_positions.clone()

        while len(remaining_key) > 0 and remaining_key[0] in node.children:
            child = node.children[remaining_key[0]]
            child.last_access_time = time.time()
            prefix_len = _key_match(child.key, remaining_key)

            if prefix_len < len(child.key):
                self._split_node(child.key, child, prefix_len)
                # After split, re-enter the loop to match the new child
                continue

            # Full match of this child node — advance past it
            mask = remaining_sel < prefix_len
            n_vals = mask.sum().item()

            remaining_key = remaining_key[prefix_len:]
            remaining_value = remaining_value[n_vals:]
            remaining_sel = remaining_sel[~mask] - prefix_len
            node = child

        if len(remaining_key) > 0:
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = remaining_key
            new_node.value = remaining_value
            new_node.position_offset = len(remaining_key) - len(remaining_value)
            new_node.selected_positions = remaining_sel.clone()
            node.children[remaining_key[0]] = new_node
            self.evictable_size_ += len(remaining_value)

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it finishes."""
        if self.disable:
            if token_ids is None:
                token_ids_len = len(req.origin_input_ids) + len(req.output_ids) - 1
            else:
                token_ids_len = len(token_ids)

            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :token_ids_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        if getattr(req, "is_kv_compressed", False):
            # Compressed KV is already stored in the tree (via insert_compressed).
            # Free only the decode-phase KV slots (those after compressed_seq_len)
            # and the request slot.  The compressed prefix slots are owned by the tree.
            decode_start = req.compressed_seq_len
            decode_end = decode_start + len(req.output_ids) - 1  # -1: first output was in prefill
            if decode_end > decode_start:
                decode_kv = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, decode_start:decode_end
                ]
                self.token_to_kv_pool_allocator.free(decode_kv)
            self.req_to_token_pool.free(req.req_pool_idx)
            if req.last_node is not None:
                self.dec_lock_ref(req.last_node)
            return

        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1]

        # When the request extended from a compressed prefix, the valid
        # KV entries in req_to_token are fewer than len(token_ids) by
        # position_offset.  Use the KV-space length to avoid reading
        # stale data, and insert_compressed to handle the mismatch.
        kv_offset = getattr(req, "_snapkv_position_offset", 0)
        kv_len = len(token_ids) - kv_offset
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len
        ]

        if kv_offset > 0:
            # Build selected_positions: the first kv_pre_len entries map
            # to scattered positions (from compressed prefix), the rest
            # are contiguous (from the extend portion).
            kv_pre_len = len(req.prefix_indices) if req.prefix_indices is not None else 0
            extend_len = kv_len - kv_pre_len
            # Prefix positions come from the tree's compressed node
            prefix_node_sel = getattr(req, "_prefix_selected_positions", None)
            if prefix_node_sel is not None and len(prefix_node_sel) == kv_pre_len:
                prefix_positions = prefix_node_sel
            else:
                prefix_positions = torch.arange(kv_pre_len, dtype=torch.int64)
            # Extend positions are contiguous starting after matched_token_len
            matched_token_len = kv_pre_len + kv_offset
            extend_positions = torch.arange(
                matched_token_len, matched_token_len + extend_len, dtype=torch.int64
            )
            all_selected = torch.cat([prefix_positions, extend_positions])
            self.insert_compressed(token_ids, kv_indices.clone(), all_selected)
        else:
            # Normal (uncompressed) path
            new_prefix_len = self.insert(token_ids, kv_indices.clone())
            self.token_to_kv_pool_allocator.free(
                kv_indices[len(req.prefix_indices) : new_prefix_len]
            )

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if token_ids is None:
            token_ids = req.fill_ids

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(token_ids)
        # For uncompressed nodes the lengths must match; for compressed
        # nodes the matched-token count (stashed on node) must match.
        matched_token_len = getattr(new_last_node, "_match_token_len", len(new_indices))
        assert matched_token_len == len(token_ids), (
            f"cache_unfinished_req: matched {matched_token_len} tokens "
            f"but expected {len(token_ids)}"
        )
        if len(new_indices) == len(token_ids):
            # Normal (uncompressed) path
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
                new_indices[len(req.prefix_indices) :],
            )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int, evict_callback: Callable):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            evict_callback(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.time()
        value = []
        selected_positions_parts = []
        matched_key_len = 0
        total_offset = 0
        while len(key) > 0 and key[0] in node.children.keys():
            child = node.children[key[0]]
            child.last_access_time = time.time()
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                if new_node.position_offset > 0 and new_node.selected_positions is not None:
                    selected_positions_parts.append(
                        new_node.selected_positions + matched_key_len
                    )
                else:
                    selected_positions_parts.append(
                        torch.arange(len(new_node.value), dtype=torch.int64) + matched_key_len
                    )
                matched_key_len += prefix_len
                total_offset += new_node.position_offset
                node = new_node
                break
            else:
                value.append(child.value)
                if child.position_offset > 0 and child.selected_positions is not None:
                    selected_positions_parts.append(
                        child.selected_positions + matched_key_len
                    )
                else:
                    selected_positions_parts.append(
                        torch.arange(len(child.value), dtype=torch.int64) + matched_key_len
                    )
                matched_key_len += len(child.key)
                total_offset += child.position_offset
                node = child
                key = key[prefix_len:]
        all_selected = (
            torch.cat([p.cpu() for p in selected_positions_parts])
            if selected_positions_parts
            else torch.tensor([], dtype=torch.int64)
        )
        return value, node, matched_key_len, total_offset, all_selected

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref

        if child.position_offset > 0:
            # Compressed node: map key-space split_len to value-space
            mask = child.selected_positions < split_len
            value_split = mask.sum().item()

            new_node.key = child.key[:split_len]
            new_node.value = child.value[:value_split]
            new_node.position_offset = split_len - value_split
            new_node.selected_positions = child.selected_positions[:value_split].clone()

            child.key = child.key[split_len:]
            child.value = child.value[value_split:]
            child.position_offset = len(child.key) - len(child.value)
            child.selected_positions = (
                child.selected_positions[value_split:] - split_len
            ).clone()
        else:
            new_node.key = child.key[:split_len]
            new_node.value = child.value[:split_len]
            child.key = child.key[split_len:]
            child.value = child.value[split_len:]

        child.parent = new_node
        new_node.parent.children[key[0]] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        total_prefix_length = 0
        while len(key) > 0 and key[0] in node.children.keys():
            node = node.children[key[0]]
            node.last_access_time = time.time()
            prefix_len = _key_match(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
            )
            for _, child in current_node.children.items():
                stack.append((child, current_indent + 2))

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.value)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list


if __name__ == "__main__":
    tree = RadixCache(None, None, False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
