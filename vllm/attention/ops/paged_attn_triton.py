"""
Copyright 2023 The vLLM team.
Copyright 2023 The BAAI team.
Copyright 2024 The SiOrigin Inc. team. All rights reserved.

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

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.attention.ops.prefix_prefill import context_attention_fwd


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]


class PagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 120, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0 and
                    blocksparse_block_size % block_size == 0), \
                (f"{blocksparse_block_size=} needs to be a multiple of"
                 f"{block_size=} used in block_tables.")

        output = torch.empty_like(query)
        block_size = value_cache.shape[3]
        paged_attention(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_dtype: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
        k_scale: float,
        v_scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        context_attention_fwd(
            query,
            key,
            value,
            output,
            kv_cache_dtype,
            key_cache,
            value_cache,
            block_tables,
            # query_start_loc is (batch_size + 1,)
            query_start_loc[:-1],
            seq_lens_tensor,
            context_lens,
            max_query_len,
            k_scale,
            v_scale,
            alibi_slopes,
            sliding_window,
        )
        return output

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        # TODO: how to implement this in triton?
        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@torch.library.custom_op("torch::swap_blocks",
                         mutates_args=["dst"],
                         device_types="cuda")
def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    """
    Copy blocks of data from `src` tensor to `dst` tensor based on the
    `block_mapping`. Supports copying between different devices (CPU <-> GPU)
    and uses non_blocking for asynchronous operations when possible.

    Args:
        src (torch.Tensor): Source tensor.
        dst (torch.Tensor): Destination tensor.
        block_mapping (torch.Tensor): Tensor of shape (N, 2). Where N is the
                                      number of block mappings. Each row
                                      specifies (src_block_idx, dst_block_idx).
    """
    # Ensure block_mapping is on the CPU
    if block_mapping.device != torch.device('cpu'):
        raise ValueError("block_mapping must be on CPU")

    # Ensure src and dst have compatible shapes for copying
    if src.size() != dst.size():
        raise ValueError(
            "Source and destination tensors must have compatible shapes.")

    # Iterate over block mappings and copy data from src to dst
    num_blocks = block_mapping.size(0)
    for i in range(num_blocks):
        src_block_number = block_mapping[i][0].item()
        dst_block_number = block_mapping[i][1].item()

        # Get the source block
        src_block = src[src_block_number]

        # Perform the block copy with non_blocking=True
        # for async operation if on different devices
        dst[dst_block_number].copy_(src_block,
                                    non_blocking=src.device != dst.device)
    if src.device != dst.device:
        torch.cuda.synchronize()


@triton.jit
def reshape_and_cache_kernel(
        key_ptr,
        value_ptr,
        key_cache_ptr,  # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache_ptr,  # [num_blocks, num_heads, head_size, block_size]
        slot_mapping_ptr,
        num_heads,
        head_size,
        block_size,
        x,
        k_scale,
        v_scale,
        stride_key,
        stride_value,
        n,
        BLOCK_SIZE: tl.constexpr):
    token_idx = tl.program_id(0)

    # Load slot mapping
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return  # Padding token, skip

    # Calculate block indices
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # Pointer to the start of the current token in key and value
    key_ptr += token_idx * stride_key
    value_ptr += token_idx * stride_value

    # Define loop over chunks of size BLOCK_SIZE
    for start in range(0, n, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        # Load keys and values for this token
        keys = tl.load(key_ptr + offsets, mask=mask)
        values = tl.load(value_ptr + offsets, mask=mask)

        # Calculate destination indices in the cache
        head_idx = offsets // head_size
        head_offset = offsets % head_size
        x_idx = head_offset // x
        x_offset = head_offset % x

        tgt_key_idx = block_idx * num_heads * (head_size // x) * \
            block_size * x + \
            head_idx * (head_size // x) * block_size * x + \
            x_idx * block_size * x + \
            block_offset * x + x_offset

        tgt_value_idx = block_idx * num_heads * head_size * block_size + \
                        head_idx * head_size * block_size + \
                        head_offset * block_size + \
                        block_offset

        # Write the scaled results to key_cache and value_cache
        tl.store(key_cache_ptr + tgt_key_idx, keys * k_scale, mask=mask)
        tl.store(value_cache_ptr + tgt_value_idx, values * v_scale, mask=mask)


def reshape_and_cache(key: torch.Tensor, value: torch.Tensor,
                      key_cache: torch.Tensor, value_cache: torch.Tensor,
                      slot_mapping: torch.Tensor, kv_cache_dtype: str,
                      k_scale: float, v_scale: float):
    assert kv_cache_dtype != "fp8", "Currently not support fp8"
    # Parameters extraction
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[3]
    x = key_cache.shape[4]

    stride_key = key.stride(0)
    stride_value = value.stride(0)
    n = num_heads * head_size

    # Define grid size and block size
    grid = (num_tokens, )
    block = min(512, n)

    # Launch Triton kernel
    reshape_and_cache_kernel[grid](key_ptr=key,
                                   value_ptr=value,
                                   key_cache_ptr=key_cache,
                                   value_cache_ptr=value_cache,
                                   slot_mapping_ptr=slot_mapping,
                                   num_heads=num_heads,
                                   head_size=head_size,
                                   block_size=block_size,
                                   x=x,
                                   k_scale=k_scale,
                                   v_scale=v_scale,
                                   stride_key=stride_key,
                                   stride_value=stride_value,
                                   n=n,
                                   BLOCK_SIZE=block)


# Requires triton >= 2.2.0
# TODO add FP8 support
@torch.library.custom_op("triton::paged_attention",
                         mutates_args=["out"],
                         device_types="cuda")
def paged_attention(
    out: torch.
    Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    query: torch.
    Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: torch.
    Tensor,  # [num_blocks, NUM_KV_HEADS, HEAD_SIZE/x, KV_BLOCK_SIZE, x]
    value_cache: torch.
    Tensor,  # [num_blocks, NUM_KV_HEADS, HEAD_SIZE, KV_BLOCK_SIZE]
    num_kv_heads: int,
    attn_scale: float,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor,  # [num_seqs]
    kv_block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: Optional[str] = None,  # TODO, support fp8
    k_scale: float = 1.0,  # TODO, support fp8
    v_scale: float = 1.0,  # TODO, support fp8
    num_splits: int = 0,
) -> None:
    assert (kv_cache_dtype == "auto") \
        , "kv_cache_dtype for fp8 is not supported now"

    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)

    x = key_cache.shape[4]
    query_group_size = num_heads // num_kv_heads

    if query_group_size == 1:
        padded_group_size = 1
    elif query_group_size < 16:
        padded_group_size = 16
    else:
        padded_group_size = triton.next_power_of_2(query_group_size)

    assert (padded_group_size == 1
            or kv_block_size >= 16), f"kv_block_size={kv_block_size}"

    # config for A100
    # TODO: support more devices and optimize
    device = torch.cuda.device_of(query)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_splits == 0:
        if num_seqs * num_kv_heads > 2 * num_sms:
            num_splits = 1
            if max_context_len >= 4096:
                partition_size = max(256, kv_block_size)
                num_splits = triton.cdiv(max_context_len, partition_size)
        else:
            partition_size = max(256, kv_block_size)
            num_splits = triton.cdiv(max_context_len, partition_size)
            if max_context_len <= 1024 or kv_block_size >= 256:
                num_splits = 1
    elif num_splits > 1:
        partition_size = triton.cdiv(max_context_len, num_splits)
        partition_size = triton.next_power_of_2(partition_size)

    with torch.cuda.device(device):
        if num_splits == 1:
            grid = (num_seqs, num_kv_heads, 1)
            _paged_attn_kernel[grid](
                out,  # dummy input
                out,  # dummy input
                out,
                query,
                key_cache,
                value_cache,
                context_lens,
                block_tables,
                alibi_slopes,
                attn_scale,
                block_tables.stride(0),
                block_tables.stride(1),
                query.stride(0),
                query.stride(1),
                query.stride(2),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                key_cache.stride(4),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                value_cache.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(1),
                out.stride(1),
                out.stride(2),
                head_size,
                padded_head_size,
                query_group_size,
                padded_group_size,
                num_kv_heads,
                kv_block_size,
                x,
                PARTITION_SIZE=0,
                USE_ALIBI_POSITION_ENCODING=alibi_slopes is not None,
            )

        else:
            grid = (num_seqs, num_kv_heads, num_splits)
            m_i = torch.empty(
                size=(num_seqs, num_kv_heads, num_splits, query_group_size),
                dtype=torch.float32,
                device=query.device,
            )
            l_i = torch.empty_like(m_i)
            tmp_out = torch.empty(
                size=(
                    num_seqs,
                    num_kv_heads,
                    num_splits,
                    query_group_size,
                    head_size,
                ),
                dtype=out.dtype,
                device=out.device,
            )

            assert (partition_size >= kv_block_size) and (
                partition_size % kv_block_size == 0
            ), f"partition_size={partition_size}, kv_block_size={kv_block_size}"
            _paged_attn_kernel[grid](
                m_i,
                l_i,
                tmp_out,
                query,
                key_cache,
                value_cache,
                context_lens,
                block_tables,
                alibi_slopes,
                attn_scale,
                block_tables.stride(0),
                block_tables.stride(1),
                query.stride(0),
                query.stride(1),
                query.stride(2),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                key_cache.stride(4),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                value_cache.stride(3),
                tmp_out.stride(0),
                tmp_out.stride(1),
                tmp_out.stride(2),
                tmp_out.stride(3),
                tmp_out.stride(4),
                head_size,
                padded_head_size,
                query_group_size,
                padded_group_size,
                num_kv_heads,
                kv_block_size,
                x,
                partition_size,
                USE_ALIBI_POSITION_ENCODING=alibi_slopes is not None,
            )

            reduce_grid = (num_seqs, num_kv_heads)
            next_num_splits = triton.next_power_of_2(num_splits)

            _paged_attn_v2_reduce_kernel[reduce_grid](
                out,
                m_i,
                l_i,
                tmp_out,
                context_lens,
                num_splits,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                head_size,
                padded_head_size,
                query_group_size,
                num_kv_heads,
                partition_size,
                next_num_splits,
            )


def get_num_warps(QUERY_GROUP_SIZE, HEAD_SIZE, KV_BLOCK_SIZE):
    if QUERY_GROUP_SIZE == 1:
        if HEAD_SIZE >= 128 and KV_BLOCK_SIZE >= 32:
            return 16
        else:
            return 8
    else:
        return 4


def get_num_stages(PARTITION_SIZE, KV_BLOCK_SIZE):
    if PARTITION_SIZE == 0:
        return 1
    else:
        if torch.cuda.get_device_capability() == (8, 0):
            if KV_BLOCK_SIZE < 256:
                return 3
            else:
                return 2
        elif torch.cuda.get_device_capability() == (8, 6):
            if KV_BLOCK_SIZE < 256:
                return 2
            else:
                return 1
        else:
            return 1


@triton.heuristics({
    "num_warps":
    lambda args: get_num_warps(args["QUERY_GROUP_SIZE"], args["HEAD_SIZE"],
                               args["KV_BLOCK_SIZE"]),
    "num_stages":
    lambda args: get_num_stages(args["QUERY_GROUP_SIZE"], args["KV_BLOCK_SIZE"]
                                ),
})
@triton.jit
def _paged_attn_kernel(
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, 
    # QUERY_GROUP_SIZE, HEAD_SIZE]
    q_ptr,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    k_cache_ptr,  # [num_blocks, NUM_KV_HEADS, HEAD_SIZE/x, KV_BLOCK_SIZE, x]
    v_cache_ptr,  # [num_blocks, NUM_KV_HEADS, HEAD_SIZE, KV_BLOCK_SIZE]
    context_lens_ptr,  # [num_seqs]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    alibi_slopes_ptr,  # [HEAD_SIZE]
    attn_scale,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_o4,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    X: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    USE_ALIBI_POSITION_ENCODING: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    part_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)

    head_idx = kv_head_idx * QUERY_GROUP_SIZE

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634

    USE_PARTITIONING = PARTITION_SIZE > 0
    context_len = tl.load(context_lens_ptr + seq_idx)
    if USE_PARTITIONING:
        context_start_idx = part_idx * PARTITION_SIZE
        if context_start_idx >= context_len:
            return
        context_end_idx = tl.minimum(context_start_idx + PARTITION_SIZE,
                                     context_len)
        num_blocks = tl.cdiv(context_end_idx - context_start_idx,
                             KV_BLOCK_SIZE)
    else:
        num_blocks = tl.cdiv(context_len, KV_BLOCK_SIZE)

    block_offset = tl.arange(0, KV_BLOCK_SIZE)
    head_offset = tl.arange(0, PADDED_HEAD_SIZE)
    padding_group_offset = tl.arange(0, PADDED_QUERY_GROUP_SIZE)

    # [HEAD_SIZE, KV_BLOCK_SIZE]
    k_offset = (kv_head_idx * stride_k_cache_h +
                (head_offset[:, None] // X) * stride_k_cache_d +
                block_offset[None, :] * stride_k_cache_bl +
                (head_offset[:, None] % X) * stride_k_cache_x)

    # [KV_BLOCK_SIZE, HEAD_SIZE]
    v_offset = (kv_head_idx * stride_v_cache_h +
                head_offset[None, :] * stride_v_cache_d +
                block_offset[:, None] * stride_v_cache_bl)

    # Load queries.
    # [PADDED_QUERY_GROUP_SIZE, HEAD_SIZE]
    q_offset = (seq_idx * stride_qbs +
                (head_idx + padding_group_offset[:, None]) * stride_qh +
                head_offset[None, :] * stride_qd)
    dim_mask = tl.where(head_offset < HEAD_SIZE, 1, 0).to(tl.int1)  # [D]
    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    # q: [PADDED_QUERY_GROUP_SIZE, HEAD_SIZE]
    q = tl.load(q_ptr + q_offset,
                mask=dim_mask[None, :] & group_mask,
                other=0.0)

    if USE_ALIBI_POSITION_ENCODING:
        alibi_slopes = tl.load(alibi_slopes_ptr + head_idx +
                               padding_group_offset[:, None],
                               mask=group_mask,
                               other=0.0)

    m_i = tl.full([PADDED_QUERY_GROUP_SIZE], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32)
    acc = tl.zeros([PADDED_QUERY_GROUP_SIZE, PADDED_HEAD_SIZE],
                   dtype=tl.float32)

    num_prev_blocks = part_idx * (PARTITION_SIZE // KV_BLOCK_SIZE)
    for i in range(num_blocks):
        block_idx = num_prev_blocks + i
        block_number = tl.load(block_tables_ptr + seq_idx * stride_b_loc_b +
                               block_idx * stride_b_loc_s)

        # Load a key block.
        k_block_offset = block_number * stride_k_cache_bs + k_offset
        v_block_offset = block_number * stride_v_cache_bs + v_offset

        k_pos_range = block_idx * KV_BLOCK_SIZE + block_offset
        k_pos_range_mask = k_pos_range < context_len
        # k_mask = mask_offset[None, :] < context_len
        # v_mask = mask_offset[:, None] < context_len

        # k: [HEAD_SIZE, KV_BLOCK_SIZE]
        k = tl.load(k_cache_ptr + k_block_offset,
                    mask=dim_mask[:, None] & k_pos_range_mask[None, :],
                    other=0.0)

        # qk: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        if PADDED_QUERY_GROUP_SIZE == 1:
            qk = tl.sum(q[:, :, None] * k[None, :, :], axis=1)
        else:
            qk = tl.dot(q, k, out_dtype=tl.float32)

        qk *= attn_scale

        if USE_ALIBI_POSITION_ENCODING:
            alibi_bias = (k_pos_range[None, :] -
                          (context_len - 1)) * alibi_slopes
            alibi_bias = tl.where(k_pos_range_mask & group_mask, alibi_bias,
                                  float("-inf"))
            qk += alibi_bias
        else:
            qk = tl.where(k_pos_range_mask, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))

        # p: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        p = tl.math.exp2((qk - m_i_new[:, None]) * log2e)
        alpha = tl.math.exp2((m_i - m_i_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLOCK_SIZE, HEAD_SIZE]
        v = tl.load(v_cache_ptr + v_block_offset,
                    mask=dim_mask[None, :] & k_pos_range_mask[:, None],
                    other=0.0)

        if PADDED_QUERY_GROUP_SIZE == 1:
            acc += tl.sum(p.T[:, :, None] * v[:, None, :], axis=0)
        else:
            p = p.to(v.dtype)
            acc += tl.dot(p, v, out_dtype=tl.float32)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    acc = acc / l_i[:, None]

    if USE_PARTITIONING:
        part_offset = ((seq_idx * NUM_KV_HEADS + kv_head_idx) *
                       max_num_partitions * QUERY_GROUP_SIZE +
                       part_idx * QUERY_GROUP_SIZE + padding_group_offset)
        mask = padding_group_offset < QUERY_GROUP_SIZE
        tl.store(m_i_ptr + part_offset, m_i, mask=mask)
        tl.store(l_i_ptr + part_offset, l_i, mask=mask)

    out_offset = seq_idx * stride_o0
    if USE_PARTITIONING:
        # tmp_out: [num_seqs, NUM_KV_HEADS, max_num_partitions,
        # QUERY_GROUP_SIZE, HEAD_SIZE]
        out_offset += kv_head_idx * stride_o1
    else:
        # out: [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
        out_offset += head_idx * stride_o1
    out_offset += (part_idx * stride_o2 +
                   padding_group_offset[:, None] * stride_o3 +
                   head_offset[None, :] * stride_o4)

    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    tl.store(out_ptr + out_offset, acc, mask=dim_mask[None, :] & group_mask)


@triton.jit
def _paged_attn_v2_reduce_kernel(
    out_ptr,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    tmp_out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions,
    # QUERY_GROUP_SIZE, HEAD_SIZE]
    context_lens_ptr,  # [num_seqs]
    max_num_partitions,  # partition stride
    stride_obs,
    stride_oh,
    stride_od,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    PADDED_NUM_PARTITIONS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    head_idx = kv_head_idx * QUERY_GROUP_SIZE

    context_len = tl.load(context_lens_ptr + seq_idx)

    head_offset = tl.arange(0, PADDED_HEAD_SIZE)

    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)
    group_head_offset = (tl.arange(0, QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE +
                         head_offset[None, :])

    dim_mask = tl.where(head_offset < HEAD_SIZE, 1, 0).to(tl.int1)  # [D]
    if num_partitions == 1:
        tmp_out_offset = ((seq_idx * NUM_KV_HEADS + kv_head_idx) *
                          max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE +
                          group_head_offset)
        tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=dim_mask[None, :])

        out_offset = (seq_idx * stride_obs + head_idx * stride_oh +
                      group_head_offset * stride_od)
        tl.store(out_ptr + out_offset, tmp_out, mask=dim_mask[None, :])
        return

    # Get the global max logit.
    ml_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions *
        QUERY_GROUP_SIZE +
        tl.arange(0, PADDED_NUM_PARTITIONS)[:, None] * QUERY_GROUP_SIZE +
        tl.arange(0, QUERY_GROUP_SIZE)[None, :])

    mask = tl.arange(0, PADDED_NUM_PARTITIONS)[:, None] < num_partitions
    # m_i: [PADDED_NUM_PARTITIONS, QUERY_GROUP_SIZE]
    m_i = tl.load(m_i_ptr + ml_offset, mask=mask, other=float("-inf"))
    # m: [QUERY_GROUP_SIZE]
    m = tl.max(m_i, axis=0)

    # Rescale the exp sums and compute the global sum.
    # l_i: [PADDED_NUM_PARTITIONS, QUERY_GROUP_SIZE]
    l_i = tl.load(l_i_ptr + ml_offset, mask=mask, other=0.0)
    l_i *= tl.exp(m_i - m[None, :])
    # l: [QUERY_GROUP_SIZE]
    l = tl.sum(l_i, axis=0)  # noqa: E741
    # r: [PADDED_NUM_PARTITIONS, QUERY_GROUP_SIZE]
    r = l_i / l[None, :]
    r = tl.reshape(r, (PADDED_NUM_PARTITIONS, QUERY_GROUP_SIZE, 1))

    tmp_out_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions *
        QUERY_GROUP_SIZE * HEAD_SIZE +
        tl.arange(0, PADDED_NUM_PARTITIONS)[:, None, None] * QUERY_GROUP_SIZE *
        HEAD_SIZE + tl.arange(0, QUERY_GROUP_SIZE)[None, :, None] * HEAD_SIZE +
        head_offset[None, None, :])
    # tmp_out: [PADDED_NUM_PARTITIONS, QUERY_GROUP_SIZE, PADDED_HEAD_SIZE]
    tmp_out = tl.load(tmp_out_ptr + tmp_out_offset,
                      mask=mask[:, :, None] & dim_mask[None, None, :],
                      other=0.0)
    # out: [QUERY_GROUP_SIZE, PADDED_HEAD_SIZE]
    out = tl.sum((tmp_out * r).to(tl.float32), axis=0)

    out_offset = (seq_idx * stride_obs + head_idx * stride_oh +
                  group_head_offset * stride_od)
    tl.store(out_ptr + out_offset, out, mask=dim_mask[None, :])
