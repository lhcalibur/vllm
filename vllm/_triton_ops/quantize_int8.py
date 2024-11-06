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

# Mapping from torch dtype to triton dtype
TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.int8: tl.int8,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}


def get_triton_dtype(torch_dtype: torch.dtype) -> tl.dtype:
    """Convert torch dtype to triton dtype."""
    if torch_dtype not in TORCH_TO_TRITON_DTYPE:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return TORCH_TO_TRITON_DTYPE[torch_dtype]

@triton.jit
def _float_to_int8_rn(x: tl.tensor) -> tl.tensor:
    """
    Convert float to int8 with round-to-nearest-even behavior using CUDA 
    libdevice.
    
    Args:
        x: Input float value
        
    Returns:
        Rounded and saturated int8 value
    """
    # Define int8 min/max constants
    I8_MIN = -128.0
    I8_MAX = 127.0

    # Convert input to float32 for consistent behavior
    x = x.to(tl.float32)

    # Round to nearest even using CUDA nearbyint function
    # This matches the behavior of std::nearbyint in the original CUDA code
    rounded = tl.extra.cuda.libdevice.nearbyint(x)

    # Saturate to int8 range
    saturated = tl.clamp(rounded, I8_MIN, I8_MAX)

    # Convert to int8
    result = saturated.to(tl.int8)

    return result


@triton.jit
def _float_to_int32_rn(x: tl.tensor) -> tl.tensor:
    """
    Convert float to int32 with round-to-nearest-even behavior using CUDA 
    libdevice.
    
    Args:
        x: Input float value
        
    Returns:
        Rounded and saturated int32 value
    """
    # Define int8 min/max constants
    I32_MIN = -2147483648.0
    I32_MAX = 2147483647.0

    # Convert input to float32 for consistent behavior
    x = x.to(tl.float32)

    # Round to nearest even using CUDA nearbyint function
    # This matches the behavior of std::nearbyint in the original CUDA code
    rounded = tl.extra.cuda.libdevice.nearbyint(x)

    # Saturate to int8 range
    # Use min/max to clamp values
    # saturated = tl.minimum(tl.maximum(rounded, I8_MIN), I8_MAX)
    saturated = tl.clamp(rounded, I32_MIN, I32_MAX)

    # Convert to int32
    result = saturated.to(tl.int32)

    return result


@triton.jit
def _int32_to_int8(x: tl.tensor) -> tl.tensor:
    # Define int8 min/max constants
    I8_MIN = -128
    I8_MAX = 127

    # Clamp the input values between int8 min/max
    x = tl.minimum(tl.maximum(x, I8_MIN), I8_MAX)

    x = x.to(tl.int8)

    return x


@triton.jit
def static_scaled_int8_quant_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,  # Can be None
    # Matrix dimensions
    hidden_size,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantize input tensor to int8 using static scaling.
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor 
        scale_ptr: Pointer to scale tensor (single value)
        azp_ptr: Pointer to zero point tensor (single value, optional)
        hidden_size: Size of hidden dimension
        BLOCK_SIZE: Size of block for parallel processing
    """
    # Program ID
    pid = tl.program_id(axis=0)

    # Compute offsets for this program instance
    offs_init = pid * hidden_size + tl.arange(0, BLOCK_SIZE)
    # Create max_offset for bounds checking
    max_offset = (pid + 1) * hidden_size

    # Load scale (single value used for entire row)
    scale = tl.load(scale_ptr)

    # Load zero point if provided
    if azp_ptr is not None:
        azp = tl.load(azp_ptr)

    # Process elements in blocks
    for offs in range(0, hidden_size, BLOCK_SIZE):
        curr_offs = offs_init + offs
        curr_mask = curr_offs < max_offset

        # Load input values
        x = tl.load(input_ptr + curr_offs, mask=curr_mask)

        # Convert to float32 for computation
        x = x.to(tl.float32)

        # Apply scaling
        x = (x / scale).to(tl.float32)

        # Add zero point if provided
        if azp_ptr is not None:
            # Round to nearest integer and add zero point
            x = _float_to_int32_rn(x) + azp
            x = _int32_to_int8(x)
        else:
            # Round to nearest integer
            x = _float_to_int8_rn(x)

        # Store result
        tl.store(output_ptr + curr_offs, x, mask=curr_mask)


@torch.library.custom_op("triton::static_scaled_int8_quant",
                         mutates_args=["out"],
                         device_types="cuda")
def static_scaled_int8_quant(out: torch.Tensor,
                             input: torch.Tensor,
                             scale: torch.Tensor,
                             azp: Optional[torch.Tensor] = None) -> None:
    """
    Quantize input tensor to int8 using static scaling.
    
    Args:
        out (torch.Tensor): Output tensor [..., hidden_size]
        input (torch.Tensor): Input tensor [..., hidden_size]
        scale (torch.Tensor): Scale tensor with single value
        azp (torch.Tensor, optional): Zero point tensor with single value
    """
    # Input validation
    assert input.is_contiguous()
    assert out.is_contiguous()
    assert scale.numel() == 1
    assert azp is None or azp.numel() == 1

    # Get dimensions
    hidden_size = input.size(-1)
    num_tokens = input.numel() // hidden_size

    # Configure meta-parameters
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 1024)

    # Configure grid
    grid = (num_tokens, )

    # Launch kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        static_scaled_int8_quant_kernel[grid](
            input_ptr=input,
            output_ptr=out,
            scale_ptr=scale,
            azp_ptr=azp,
            hidden_size=hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )


@triton.jit
def dynamic_scaled_int8_quant_kernel(
        input_ptr,  # Input tensor pointer (fp16/bf16/fp32)
        output_ptr,  # Output tensor pointer (int8)
        scale_ptr,  # Scale tensor pointer (fp32)
        hidden_size,  # Hidden dimension size
        BLOCK_SIZE: tl.constexpr,  # Triton block size
):
    # Get program ID
    pid = tl.program_id(0)

    # Calculate current token offset
    token_offset = pid * hidden_size + tl.arange(0, BLOCK_SIZE)
    max_token_offset = (pid + 1) * hidden_size

    # Load data and calculate maximum absolute value
    max_val = tl.cast(float('-inf'), tl.float32)
    for start_idx in range(0, hidden_size, BLOCK_SIZE):
        # Load input data
        offset = token_offset + start_idx
        mask = offset < max_token_offset
        x = tl.load(input_ptr + offset, mask=mask, other=0.0)

        # Calculate absolute value and update maximum
        x_abs = tl.abs(x)
        max_val = tl.maximum(max_val, tl.max(x_abs, axis=0).to(tl.float32))

    # Calculate scale and store
    scale = max_val / 127.0
    tl.store(scale_ptr + pid, scale)

    # Quantize data
    scale_recip = 127.0 / max_val
    for start_idx in range(0, hidden_size, BLOCK_SIZE):
        offset = token_offset + start_idx
        mask = offset < max_token_offset
        x = tl.load(input_ptr + offset, mask=mask)

        # Convert to float32 for computation
        x = x.to(tl.float32)

        # Quantize to int8
        x_scaled = x * scale_recip
        x_int8 = _float_to_int8_rn(x_scaled)

        tl.store(output_ptr + offset, x_int8, mask=mask)


@triton.jit
def dynamic_scaled_int8_azp_quant_kernel(
    input_ptr,  # Input tensor pointer
    output_ptr,  # Output tensor pointer
    scale_ptr,  # Scale tensor pointer
    azp_ptr,  # Zero point tensor pointer
    hidden_size,  # Hidden dimension size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offset = pid * hidden_size + tl.arange(0, BLOCK_SIZE)
    max_token_offset = (pid + 1) * hidden_size

    # First pass: Find min/max values
    min_val = tl.cast(float('inf'), tl.float32)
    max_val = tl.cast(float('-inf'), tl.float32)

    for start_idx in range(0, hidden_size, BLOCK_SIZE):
        offset = token_offset + start_idx
        mask = offset < max_token_offset
        x = tl.load(input_ptr + offset, mask=mask)

        min_val = tl.minimum(
            min_val,
            tl.min(tl.where(mask, x, tl.cast(float('inf'), x.dtype)),
                   axis=0).to(tl.float32))
        max_val = tl.maximum(
            max_val,
            tl.max(tl.where(mask, x, tl.cast(float('-inf'), x.dtype)),
                   axis=0).to(tl.float32))

    # Calculate scale and zero point
    scale = (max_val - min_val) / 255.0
    azp = tl.extra.cuda.libdevice.nearbyint(-128.0 - min_val / scale)

    # Store scale and zero point
    tl.store(scale_ptr + pid, scale)
    tl.store(azp_ptr + pid, tl.cast(azp, tl.int32))

    # Second pass: Quantize data
    for start_idx in range(0, hidden_size, BLOCK_SIZE):
        offset = token_offset + start_idx
        mask = offset < max_token_offset
        x = tl.load(input_ptr + offset, mask=mask)

        # Convert to float32 for computation
        x = x.to(tl.float32)
        # Apply scaling
        x = (x / scale).to(tl.float32)

        # Round to nearest integer and add zero point
        x_int32 = _float_to_int32_rn(x) + azp
        x_int8 = _int32_to_int8(x_int32)

        tl.store(output_ptr + offset, x_int8, mask=mask)


@torch.library.custom_op("triton::dynamic_scaled_int8_quant",
                         mutates_args=["out", "scales", "azp"],
                         device_types="cuda")
def dynamic_scaled_int8_quant(out: torch.Tensor,
                              input: torch.Tensor,
                              scales: torch.Tensor,
                              azp: Optional[torch.Tensor] = None) -> None:
    assert input.is_contiguous()
    assert out.is_contiguous()
    assert scales.is_contiguous()
    assert azp is None or azp.is_contiguous()

    hidden_size = input.size(-1)
    num_tokens = input.numel() // hidden_size
    # Configure meta-parameters
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 1024)

    grid = (num_tokens, )
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        if azp is None:
            dynamic_scaled_int8_quant_kernel[grid](
                input_ptr=input,
                output_ptr=out,
                scale_ptr=scales,
                hidden_size=hidden_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            dynamic_scaled_int8_azp_quant_kernel[grid](
                input_ptr=input,
                output_ptr=out,
                scale_ptr=scales,
                azp_ptr=azp,
                hidden_size=hidden_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized 
    tensor, scale, and optionally azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp 
            ignored).

    Returns:
      Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: Output int8
      scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp is
            None), "azp must only be provided for asymmetric quantization."
        static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, None

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    dynamic_scaled_int8_quant(output, input, input_scales, input_azp)
    return output, input_scales, input_azp


def get_scaled_mm_autotune_config():
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=5,
            num_warps=2),
        # Good config for fp8 inputs.
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4)
    ]


@triton.autotune(
    configs=get_scaled_mm_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def scaled_mm_kernel(
        # Output pointer
        c_ptr,
        # Input pointers
        a_ptr,
        b_ptr,
        # Pointers to scales and zero points
        a_scales_ptr,
        b_scales_ptr,
        azp_adj_ptr,
        azp_ptr,
        # Bias pointer (optional)
        bias_ptr,
        # Matrix dimensions
        M,
        N,
        K,
        # Whether to use bias
        HAS_BIAS: tl.constexpr,
        # Stride information
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Block sizes
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        A_SCALE_ROWS: tl.constexpr,  # True if a_scales is [M], False if [1]
        B_SCALE_COLS: tl.constexpr,  # True if b_scales is [N], False if [1]
        HAS_AZP_ADJ: tl.constexpr,  # True if azp_adj is not None
        HAS_AZP: tl.constexpr,  # True if azp is not None
        OUT_DTYPE: tl.constexpr,  # Output data type
):
    # Program ID
    pid = tl.program_id(axis=0)
    # Number of program ids along the M axis
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # Number of programs ids along the N axis
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Number of programs in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # Id of the group this program is in
    group_id = pid // num_pid_in_group
    # Row-id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is
    # smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # *Within groups*, programs are ordered in a column-major order
    # Row-id of the program in the *launch grid*
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # Col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offset_m = pid_m * BLOCK_SIZE_M
    offset_n = pid_n * BLOCK_SIZE_N
    offs_am = (offset_m + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (offset_n + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create pointer blocks for matrices
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of int32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K
        # dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs,
                    mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                    other=0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0)

        # Accumulate matrix multiplication in int32
        accumulator += tl.dot(a, b, out_dtype=tl.int32)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Handle AZP correction
    if HAS_AZP_ADJ:
        if HAS_AZP:
            # Per-token AZP case
            azp = tl.load(azp_ptr + offs_am).to(
                dtype=tl.int32)  # Shape: [BLOCK_SIZE_M] dtype=int32
            azp_adj = tl.load(azp_adj_ptr + offs_bn).to(
                dtype=tl.int32)  # Shape: [BLOCK_SIZE_N] dtype=int32
            azp_correction = azp[:, None] * azp_adj[None, :]
        else:
            # Per-tensor AZP case
            azp_correction = tl.load(azp_adj_ptr +
                                     offs_bn)  # Shape: [BLOCK_SIZE_N]
        # Subtract AZP correction
        accumulator = accumulator - azp_correction

    # Post-processing
    # Convert to float and apply scales
    accumulator = accumulator.to(tl.float32)

    # Load scaling factors
    # Handle per-row scaling for matrix A
    if A_SCALE_ROWS:
        a_scale_ptr = a_scales_ptr + offs_am[:, None]
        a_scales = tl.load(a_scale_ptr)
    else:
        a_scales = tl.load(a_scales_ptr)  # Single scale for all rows

    # Handle per-column scaling for matrix B
    if B_SCALE_COLS:
        b_scale_ptr = b_scales_ptr + offs_bn[None, :]
        b_scales = tl.load(b_scale_ptr)
    else:
        b_scales = tl.load(b_scales_ptr)  # Single scale for all columns

    # Scale the accumulated values
    accumulator *= a_scales * b_scales
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn[None, :])
        # Bias is in the same dtype as the output
        # Convert bias to float32
        bias = bias.to(dtype=tl.float32)
        accumulator += bias

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = offset_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = offset_n + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:,
                                         None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    accumulator = accumulator.to(dtype=OUT_DTYPE)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# TODO support FP8
@torch.library.custom_op("triton::scaled_mm",
                         mutates_args=["out"],
                         device_types="cuda")
def scaled_mm(out: torch.Tensor,
              a: torch.Tensor,
              b: torch.Tensor,
              a_scales: torch.Tensor,
              b_scales: torch.Tensor,
              azp_adj: Optional[torch.Tensor] = None,
              azp: Optional[torch.Tensor] = None,
              bias: Optional[torch.Tensor] = None) -> None:
    """
    Custom CUDA kernel for quantized matrix multiplication with scaling.
    Computes: out = (a_scales * A) @ (b_scales * B) + bias
    
    Args:
        out: Output tensor to store results
        a: Input matrix A (quantized)
        b: Input matrix B (quantized) 
        a_scales: Scaling factors for A
        b_scales: Scaling factors for B
        bias: Optional bias tensor
    """
    # Get matrix dimensions
    # M: number of rows in A (batch size)
    # K: number of columns in A / rows in B (hidden dimension)
    # N: number of columns in B (output dimension)
    assert b.stride(0) == 1, "b must have contiguous K dimension"

    M, K = a.shape
    _, N = b.shape

    # Calculate grid size to cover entire output matrix
    # Each block computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of the output
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
        N, META['BLOCK_SIZE_N']), )

    # Determine if we have per-row/per-column scaling
    # a_scale_rows: True if we have different scale per row of A
    # b_scale_cols: True if we have different scale per column of B
    a_scale_rows = (a_scales.numel() == M)
    b_scale_cols = (b_scales.numel() == N)

    # Launch the CUDA kernel on the appropriate device
    device = torch.cuda.device_of(a)
    with torch.cuda.device(device):
        scaled_mm_kernel[grid](
            out,
            a,
            b,
            a_scales,
            b_scales,
            azp_adj if azp_adj is not None else
            a,  # Use a as placeholder if no azp_adj
            azp if azp is not None else a,  # Use a as placeholder if no azp
            bias if bias is not None else a,  # Use a as placeholder if no bias
            M,
            N,
            K,
            bias is not None,  # HAS_BIAS flag
            a.stride(0),
            a.stride(1),  # Strides for matrix A
            b.stride(0),
            b.stride(1),  # Strides for matrix B
            out.stride(0),
            out.stride(1),  # Strides for output matrix
            A_SCALE_ROWS=a_scale_rows,  # Whether A has per-row scaling
            B_SCALE_COLS=b_scale_cols,  # Whether B has per-column scaling
            HAS_AZP=azp is not None,  # Whether AZP is present
            HAS_AZP_ADJ=azp_adj is not None,  # Whether AZP_ADJ is present
            OUT_DTYPE=get_triton_dtype(out.dtype),  # Output data type
        )


def triton_scaled_mm(a: torch.Tensor,
                     b: torch.Tensor,
                     a_scales: torch.Tensor,
                     b_scales: torch.Tensor,
                     out_dtype: torch.dtype,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs quantized matrix multiplication: 
    C = (a_scales * A) @ (b_scales * B) + bias
    
    Args:
        a: Input matrix A (int8/fp8) of shape [M, K]
        b: Input matrix B (int8/fp8) of shape [K, N] 
        a_scales: Quantization scales for A (fp16/bf16) of shape [M] or [1]
        b_scales: Quantization scales for B (fp16/bf16) of shape [N] or [1]
        out_dtype: Output data type (fp16/bf16)
        bias: Optional bias tensor (fp16/bf16) of shape [N]

    Returns:
        c: Output matrix (fp16/bf16) of shape [M, N]
    """
    # Get matrix dimensions
    M, K = a.shape  # M: batch size, K: hidden dimension
    _, N = b.shape  # N: output dimension

    # Initialize output tensor with specified dtype and device
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # Call the custom CUDA kernel for scaled matrix multiplication
    scaled_mm(c, a, b, a_scales, b_scales, bias=bias)

    return c


def triton_scaled_mm_azp(a: torch.Tensor,
                         b: torch.Tensor,
                         a_scales: torch.Tensor,
                         b_scales: torch.Tensor,
                         out_dtype: torch.dtype,
                         azp_adj: torch.Tensor,
                         azp: Optional[torch.Tensor] = None,
                         bias: Optional[torch.Tensor] = None):
    """
    Wrapper function for the Triton kernel implementing scaled matrix
    multiplication with AZP.
    
    Computes: C = (a_scales * (A - azp)) @ (b_scales * B) + bias
    where azp_adj = sum(B, dim=0) is precomputed for efficiency
    """
    # Validate input tensor data types
    assert b.stride(0) == 1, "b must have contiguous K dimension"
    assert a.dtype in [torch.int8, torch.float8_e4m3fn]
    assert b.dtype == a.dtype
    assert a_scales.dtype == torch.float32
    assert b_scales.dtype == torch.float32

    # Extract matrix dimensions
    # M = batch size, K = hidden dim, N = output dim
    M, K = a.shape
    _, N = b.shape

    # Allocate output tensor with specified dtype
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # Step 5: Call Triton kernel for scaled matrix multiply with AZP
    scaled_mm(c, a, b, a_scales, b_scales, azp_adj, azp, bias)

    # Return computed result
    return c
