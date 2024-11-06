"""
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
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_rotary_embedding_kernel(
        query_ptr,  # [batch_size, seq_len, num_heads * head_size] or
        # [num_tokens, num_heads * head_size]
    key_ptr,  # [batch_size, seq_len, num_kv_heads * head_size] or
        # [num_tokens, num_kv_heads * head_size]
    cos_sin_cache_ptr,  # [max_position, rot_dim]
        rot_dim,  # rotation dimension
        query_token_offset,
        key_token_offset,
        num_heads,  # number of attention heads
        num_kv_heads,  # number of key/value heads
        head_size,  # size of each head
        is_neox: tl.constexpr,  # whether to use NeoX style rotation
        BLOCK_SIZE: tl.constexpr,  # must be power of 2
):
    # Calculate embedding dimension and cache offsets
    embed_dim = rot_dim // 2

    # Calculate the total number of elements to process for queries
    nq = num_heads * embed_dim
    for i in range(0, nq, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)

        head_idx = idx // embed_dim
        rot_offset = idx % embed_dim

        mask = idx < nq

        # Load cos and sin values
        cos = tl.load(cos_sin_cache_ptr + rot_offset, mask=mask)
        sin = tl.load(cos_sin_cache_ptr + embed_dim + rot_offset, mask=mask)

        # Calculate offsets for x1 and x2
        x_offset = query_token_offset + head_idx * head_size

        if is_neox:
            x1_offset = x_offset + rot_offset
            x2_offset = x_offset + rot_offset + embed_dim
        else:
            x1_offset = x_offset + 2 * rot_offset
            x2_offset = x_offset + 2 * rot_offset + 1

        # Load and transform query values
        x1 = tl.load(query_ptr + x1_offset, mask=mask)
        x2 = tl.load(query_ptr + x2_offset, mask=mask)

        # Apply rotation
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        # Store rotated values
        tl.store(query_ptr + x1_offset, out1, mask=mask)
        tl.store(query_ptr + x2_offset, out2, mask=mask)

    # Process key heads
    num_key_blocks = (num_kv_heads * embed_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    for block_idx in range(num_key_blocks):
        offset = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE is power of 2
        idx = block_idx * BLOCK_SIZE + offset

        head_idx = idx // embed_dim
        rot_offset = idx % embed_dim

        mask = idx < (num_kv_heads * embed_dim)

        cos = tl.load(cos_sin_cache_ptr + rot_offset, mask=mask)
        sin = tl.load(cos_sin_cache_ptr + embed_dim + rot_offset, mask=mask)

        x_offset = key_token_offset + head_idx * head_size

        if is_neox:
            x1_offset = x_offset + rot_offset
            x2_offset = x_offset + rot_offset + embed_dim
        else:
            x1_offset = x_offset + 2 * rot_offset
            x2_offset = x_offset + 2 * rot_offset + 1

        x1 = tl.load(key_ptr + x1_offset, mask=mask)
        x2 = tl.load(key_ptr + x2_offset, mask=mask)

        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        tl.store(key_ptr + x1_offset, out1, mask=mask)
        tl.store(key_ptr + x2_offset, out2, mask=mask)


@triton.jit
def rotary_embedding_kernel(
        positions_ptr,  # [batch_size, seq_len] or [num_tokens]
        query_ptr,  # [batch_size, seq_len, num_heads * head_size] or
        # [num_tokens, num_heads * head_size]
    key_ptr,  # [batch_size, seq_len, num_kv_heads * head_size] or
        # [num_tokens, num_kv_heads * head_size]
    cos_sin_cache_ptr,  # [max_position, rot_dim]
        num_tokens,  # number of total tokens
        rot_dim,  # rotation dimension
        query_stride,  # stride for query tensor
        key_stride,  # stride for key tensor
        num_heads,  # number of attention heads
        num_kv_heads,  # number of key/value heads
        head_size,  # size of each head
        is_neox: tl.constexpr,  # whether to use NeoX style rotation
        BLOCK_SIZE: tl.constexpr,  # must be power of 2
):
    pid = tl.program_id(0)  # token index

    # Exit if the program ID is beyond the number of tokens
    if pid >= num_tokens:
        return

    # Load position for current token
    pos = tl.load(positions_ptr + pid)

    # Calculate pointers for the current token's query and key
    cache_ptr = cos_sin_cache_ptr + pos * rot_dim
    query_token_offset = pid * query_stride
    key_token_offset = pid * key_stride

    _apply_rotary_embedding_kernel(query_ptr, key_ptr, cache_ptr, rot_dim,
                                   query_token_offset, key_token_offset,
                                   num_heads, num_kv_heads, head_size, is_neox,
                                   BLOCK_SIZE)


@torch.library.custom_op("triton::rotary_embedding",
                         mutates_args=["query", "key"],
                         device_types="cuda")
def rotary_embedding(
    positions: torch.Tensor,  # [batch_size, seq_len] or [num_tokens]
    query: torch.Tensor,  # [batch_size, seq_len, num_heads * head_size] 
    # or [num_tokens, num_heads * head_size]
    key: torch.Tensor,  # [batch_size, seq_len, num_kv_heads * head_size] 
    # or [num_tokens, num_kv_heads * head_size]
    head_size: int,
    cos_sin_cache: torch.Tensor,  # [max_position, rot_dim]
    is_neox: bool = False,
) -> None:
    # Get dimensions
    num_tokens = query.numel() // query.size(-1)
    rot_dim = cos_sin_cache.size(1)
    num_heads = query.size(-1) // head_size
    num_kv_heads = key.size(-1) // head_size
    query_stride = query.stride(-2)
    key_stride = key.stride(-2)

    # Calculate optimal block size (similar to CUDA implementation's approach)
    BLOCK_SIZE = min(triton.next_power_of_2(num_heads * rot_dim // 2), 512)

    # Launch kernel
    grid = (num_tokens, )
    device = torch.cuda.device_of(query)
    with torch.cuda.device(device):
        rotary_embedding_kernel[grid](
            positions,
            query,
            key,
            cos_sin_cache,
            num_tokens,
            rot_dim,
            query_stride,
            key_stride,
            num_heads,
            num_kv_heads,
            head_size,
            is_neox,
            BLOCK_SIZE,
        )


@triton.jit
def batched_rotary_embedding_kernel(
    # Pointers to matrices
    positions_ptr,  # [num_tokens]
    query_ptr,  # [num_tokens, num_heads, head_size]
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    cos_sin_cache_ptr,  # [max_position, 2, rot_dim // 2]
    cos_sin_cache_offsets_ptr,  # [num_tokens]
    num_tokens,
    # Matrix dimensions
    rot_dim,
    query_stride,
    key_stride,
    num_heads,
    num_kv_heads,
    head_size,
    is_neox: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID - each thread block processes one token
    pid = tl.program_id(0)

    # Exit if the program ID is beyond the number of tokens
    if pid >= num_tokens:
        return

    # Compute the offsets for this token
    pos = tl.load(positions_ptr + pid)
    cos_sin_cache_offset = tl.load(cos_sin_cache_offsets_ptr + pid)
    # Calculate pointers for the current token's query and key
    cache_offset = (cos_sin_cache_offset + pos) * rot_dim
    cache_ptr = cos_sin_cache_ptr + cache_offset

    # Load the rotation matrices for this position
    query_token_offset = pid * query_stride
    key_token_offset = pid * key_stride

    _apply_rotary_embedding_kernel(query_ptr, key_ptr, cache_ptr, rot_dim,
                                   query_token_offset, key_token_offset,
                                   num_heads, num_kv_heads, head_size, is_neox,
                                   BLOCK_SIZE)


@torch.library.custom_op("triton::batched_rotary_embedding",
                         mutates_args=["query", "key"],
                         device_types="cuda")
def batched_rotary_embedding(
        positions: torch.Tensor,  # [num_tokens]
        query: torch.Tensor,  # [num_tokens, num_heads * head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads * head_size]
        head_size: int,
        cos_sin_cache: torch.Tensor,  # [max_position, rot_dim]
        is_neox: bool,
        rot_dim: int,
        cos_sin_cache_offsets: torch.Tensor,  # [num_tokens]
) -> None:
    num_tokens = cos_sin_cache_offsets.size(0)
    num_heads = query.size(-1) // head_size
    num_kv_heads = key.size(-1) // head_size
    query_stride = query.stride(-2)
    key_stride = key.stride(-2)

    # Match CUDA implementation's block size: min(num_heads * rot_dim / 2, 512)
    BLOCK_SIZE = min(triton.next_power_of_2(num_heads * rot_dim // 2), 512)

    # Match CUDA implementation's grid configuration: dim3(num_tokens)
    grid = (num_tokens, )
    device = torch.cuda.device_of(query)
    with torch.cuda.device(device):
        batched_rotary_embedding_kernel[grid](
            positions,
            query,
            key,
            cos_sin_cache,
            cos_sin_cache_offsets,
            num_tokens,
            rot_dim,
            query_stride,
            key_stride,
            num_heads,
            num_kv_heads,
            head_size,
            is_neox,
            BLOCK_SIZE=BLOCK_SIZE,
        )

