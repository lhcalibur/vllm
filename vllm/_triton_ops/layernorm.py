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
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["epsilon"])
def rms_norm_kernel(out_ptr, input_ptr, weight_ptr, hidden_size, epsilon,
                    BLOCK_SIZE: tl.constexpr):
    # Compute block and thread indices
    block_id = tl.program_id(0)  # Each block handles one token
    idx = tl.arange(0, BLOCK_SIZE)  # Per-thread index

    # Start the reduction at zero
    sum_squares = tl.zeros((), dtype=tl.float32)  # Accumulate as a scalar

    # Loop over chunks of the hidden_size
    for offset in range(0, hidden_size, BLOCK_SIZE):
        current_idx = idx + offset
        mask = current_idx < hidden_size

        # Load input data for the current chunk
        input_val = tl.load(input_ptr + block_id * hidden_size + current_idx,
                            mask=mask).to(tl.float32)

        # Compute squared values and accumulate for variance
        sq_input_val = input_val * input_val
        sum_squares += tl.sum(tl.where(mask, sq_input_val, 0.0), axis=0)

    # Compute the RMS (Root Mean Square) normalization factor
    variance = sum_squares / hidden_size
    norm_factor = tl.rsqrt(variance + epsilon)

    # Loop again to normalize and apply weight
    for offset in range(0, hidden_size, BLOCK_SIZE):
        current_idx = idx + offset
        mask = current_idx < hidden_size

        # Load input data and weight for the current chunk
        input_val = tl.load(input_ptr + block_id * hidden_size + current_idx,
                            mask=mask)
        weight = tl.load(weight_ptr + current_idx, mask=mask)

        # Normalize input and apply weight
        normalized_input = (input_val.to(tl.float32) * norm_factor).to(
            input_val.dtype) * weight

        # Store the result
        tl.store(out_ptr + block_id * hidden_size + current_idx,
                 normalized_input,
                 mask=mask)


@torch.library.custom_op("triton::rms_norm",
                         mutates_args=["out"],
                         device_types="cuda")
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    # Extract shapes
    num_tokens, hidden_size = input.shape

    # Define the grid/block size for Triton
    BLOCK_SIZE = min(
        triton.next_power_of_2(hidden_size),
        1024)  # A fixed block size for processing chunks of hidden_size
    grid = (num_tokens, )

    # Launch the kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        rms_norm_kernel[grid](out, input, weight, hidden_size, epsilon,
                              BLOCK_SIZE)


@triton.jit(do_not_specialize=["epsilon"])
def fused_add_rms_norm_kernel(input_ptr, residual_ptr, weight_ptr, hidden_size,
                              epsilon, BLOCK_SIZE: tl.constexpr):
    # Block and thread indexing
    block_id = tl.program_id(0)  # Each block handles one token
    idx = tl.arange(0, BLOCK_SIZE)  # Per-thread index

    # Initialize sum of squares accumulator
    sum_squares = tl.zeros((), dtype=tl.float32)

    # First pass: Compute sum of squares and fuse addition
    for offset in range(0, hidden_size, BLOCK_SIZE):
        current_idx = idx + offset
        mask = current_idx < hidden_size

        # Load input and residual, perform the fused addition
        input_val = tl.load(input_ptr + block_id * hidden_size + current_idx,
                            mask=mask)
        residual_val = tl.load(residual_ptr + block_id * hidden_size +
                               current_idx,
                               mask=mask)
        fused_val = input_val + residual_val
        tl.store(residual_ptr + block_id * hidden_size + current_idx,
                 fused_val,
                 mask=mask)

        # Accumulate sum of squares for variance calculation
        fused_val = tl.cast(fused_val, tl.float32)
        sum_squares += tl.sum(tl.where(mask, fused_val * fused_val, 0.0),
                              axis=0)

    # Compute variance and RMS normalization factor
    variance = sum_squares / hidden_size
    norm_factor = tl.rsqrt(variance + epsilon)

    # Second pass: Apply normalization and weight
    for offset in range(0, hidden_size, BLOCK_SIZE):
        current_idx = idx + offset
        mask = current_idx < hidden_size

        # Load fused values and weight, normalize and scale
        x = tl.load(residual_ptr + block_id * hidden_size + current_idx,
                    mask=mask)
        weight = tl.load(weight_ptr + current_idx, mask=mask)
        normalized_input = (x.to(tl.float32) * norm_factor).to(
            x.dtype) * weight

        # Store the result in the output tensor
        tl.store(input_ptr + block_id * hidden_size + current_idx,
                 normalized_input,
                 mask=mask)


# Triton custom op for fused_add_rms_norm
@torch.library.custom_op("triton::fused_add_rms_norm",
                         mutates_args=["input", "residual"],
                         device_types="cuda")
def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    # Get dimensions
    num_tokens, hidden_size = input.shape

    # Define grid and block size
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 1024)
    grid = (num_tokens, )

    # Launch the kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        fused_add_rms_norm_kernel[grid](input, residual, weight, hidden_size,
                                        epsilon, BLOCK_SIZE)

