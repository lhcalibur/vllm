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

SILU_FUNC_TYPE: tl.constexpr = 0
GELU_FUNC_TYPE: tl.constexpr = 1
GELU_TANH_FUNC_TYPE: tl.constexpr = 2
GELU_NEW_FUNC_TYPE: tl.constexpr = 3
GELU_FAST_FUNC_TYPE: tl.constexpr = 4
GELU_QUICK_FUNC_TYPE: tl.constexpr = 5


@triton.jit
def _silu_kernel(x):
    # SiLU (x * sigmoid(x))

    # # Use 2^x instead of e^x results in an additional scale factor of log2(e)
    # return x / (1.0 + tl.exp2(-x * 1.44269504089))

    dtype = x.dtype
    x = tl.cast(x, tl.float32)
    return (x * tl.sigmoid(x)).to(dtype)


@triton.jit
def _gelu_kernel(x):
    # GELU kernel based on CUAD code
    # Equivalent to PyTorch GELU with 'none' approximation
    M_SQRT1_2 = 0.70710678118654752440  # 1/sqrt(2)
    dtype = x.dtype
    x = tl.cast(x, tl.float32)
    return (x * 0.5 * (1.0 + tl.erf(x * M_SQRT1_2))).to(dtype)


@triton.jit
def _tanhf(x):
    # Tanh is just a scaled sigmoid
    CONST_2F = tl.cast(2, tl.float32)
    x = tl.cast(x, tl.float32)
    return CONST_2F * tl.sigmoid(CONST_2F * x) - tl.cast(1, tl.float32)


@triton.jit
def _gelu_tanh_kernel(x):
    # GELU kernel based on 'tanh' approximation
    # Equivalent to PyTorch GELU with 'tanh' approximation
    M_SQRT2 = 1.41421356237309504880  # sqrt(2)
    M_2_SQRTPI = 1.12837916709551257390  # 2/sqrt(pi)
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715

    dtype = x.dtype
    x = tl.cast(x, tl.float32)

    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return (x * 0.5 * (1.0 + _tanhf(inner))).to(dtype)


@triton.jit
def _gelu_new_kernel(x):
    # GELU kernel based on the 'new' approximation
    # Equivalent to the CUDA code provided
    x_cube_f = (x * x * x).to(tl.float32)
    inner = tl.cast(0.79788456, tl.float32) * (
        x +
        (tl.cast(0.044715, tl.float32) * x_cube_f).to(x.dtype)).to(tl.float32)
    t = _tanhf(inner.to(x.dtype)).to(x.dtype)
    return tl.cast(0.5, x.dtype) * x * (tl.cast(1.0, x.dtype) + t)


@triton.jit
def _gelu_fast_kernel(x):
    # GELU kernel based on the 'fast' approximation
    # Similar to the CUDA code for fast approximation
    x_f = tl.cast(x, tl.float32)
    t = _tanhf((tl.cast(0.79788456, tl.float32) * x_f).to(x.dtype) *
               (tl.cast(1.0, x.dtype) +
                (tl.cast(0.044715, tl.float32) * x_f).to(x.dtype) * x))
    t = t.to(x.dtype)
    return tl.cast(0.5, x.dtype) * x * (tl.cast(1.0, x.dtype) + t)


@triton.jit
def _gelu_quick_kernel(x):
    # GELU kernel based on the 'quick' approximation
    # Equivalent to x * sigmoid(1.702 * x)
    # Use 2^x instead of e^x results in an additional scale factor of log2(e)
    # return x / (1.0 + tl.exp2(-1.702 * x * 1.44269504089))

    dtype = x.dtype
    x_f = tl.cast(x, tl.float32)
    return (x_f * tl.sigmoid(tl.cast(1.702, tl.float32) * x_f)).to(dtype)


@triton.jit
def act_and_mul_kernel(
    out_ptr,  # Output pointer (flattened tensor)
    input_ptr,  # Input pointer (flattened tensor)
    d: int,  # Dimension size
    BLOCK_SIZE: tl.constexpr,  # Block size (threads per block)
    ACTIVATION_TYPE: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    # Program IDs
    # Cast to int64 to prevent from Invalid memory access:
    # https://github.com/triton-lang/triton/issues/1058
    token_id = tl.program_id(0).to(
        tl.int64
        if USE_INT64 else tl.int32)  # Get token index for this kernel instance
    block_start = tl.arange(
        0, BLOCK_SIZE)  # Create a range of BLOCK_SIZE elements

    # Precompute common base pointers outside the loop for efficiency
    input_base_ptr1 = input_ptr + token_id * 2 * d
    input_base_ptr2 = input_ptr + token_id * 2 * d + d
    out_ptr_base = out_ptr + token_id * d

    # Iterate over all blocks in the dimension to process all elements
    for i in range(0, d, BLOCK_SIZE):
        block_id = block_start + i

        # Mask to ensure we do not read/write out of bounds
        mask = block_id < d

        # Calculate pointers for the two parts of the input:
        # input and gating values
        input_ptr1 = input_base_ptr1 + block_id  # First half (input values)
        input_ptr2 = input_base_ptr2 + block_id  # Second half (gate values)

        # Load input values with masking
        input_vals = tl.load(input_ptr1, mask=mask)  # Load input values

        # Apply the selected activation function on input_vals
        if ACTIVATION_TYPE == SILU_FUNC_TYPE:
            activated_vals = _silu_kernel(input_vals)
        elif ACTIVATION_TYPE == GELU_FUNC_TYPE:
            activated_vals = _gelu_kernel(input_vals)
        elif ACTIVATION_TYPE == GELU_TANH_FUNC_TYPE:
            activated_vals = _gelu_tanh_kernel(input_vals)
        else:
            activated_vals = input_vals  # No activation if the type is unknown

        # Load values with masking
        gate_vals = tl.load(input_ptr2, mask=mask)  # Load gate values
        # Elementwise multiply activation with gate_vals
        result = activated_vals.to(gate_vals.dtype) * gate_vals

        # Store the result in the output tensor
        tl.store(out_ptr_base + block_id, result, mask=mask)


def launch_activation_and_mul(out: torch.Tensor, input: torch.Tensor,
                              activation_type: int) -> None:
    """
    Applies activation function to the first half of `input` and
    multiplies it element-wise with the second half of `input`, storing the
    result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., 2 * d].
    """
    # Assuming input has shape [..., 2 * d] and out has shape [..., d]
    d: int = input.shape[
        -1] // 2  # The dimension of the output (half of input's last dim)
    num_tokens: int = input.numel() // input.shape[-1]  # Number of tokens

    # Define block size (maximum threads per block is typically 1024 in Triton)
    BLOCK_SIZE: int = min(triton.next_power_of_2(d), 1024)

    # Check if we need to use int64 for indexing.
    use_int64 = input.numel() > 2**31 - 1

    # Launch the Triton kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        act_and_mul_kernel[(num_tokens, )](
            out,
            input,
            d,
            BLOCK_SIZE=BLOCK_SIZE,
            ACTIVATION_TYPE=activation_type,
            USE_INT64=use_int64,
        )


@torch.library.custom_op("triton::silu_and_mul",
                         mutates_args=["out"],
                         device_types="cuda")
def silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    """
    Applies the SiLU activation function to the first half of `input` and
    multiplies it element-wise with the second half of `input`, storing the
    result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., 2 * d].
    """
    launch_activation_and_mul(out, input, SILU_FUNC_TYPE)


@torch.library.custom_op("triton::gelu_and_mul",
                         mutates_args=["out"],
                         device_types="cuda")
def gelu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    """
    Applies the GeLU activation function to the first half of `input` and
    multiplies it element-wise with the second half of `input`, storing the
    result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., 2 * d].
    """
    launch_activation_and_mul(out, input, GELU_FUNC_TYPE)


@torch.library.custom_op("triton::gelu_tanh_and_mul",
                         mutates_args=["out"],
                         device_types="cuda")
def gelu_tanh_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    """
    Applies the GeLU activation function with 'tanh' approximation to the first
    half of `input` and multiplies it element-wise with the second half of
    `input`, storing the result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., 2 * d].
    """
    launch_activation_and_mul(out, input, GELU_TANH_FUNC_TYPE)


@triton.jit
def activation_kernel(
    out_ptr,  # Output pointer (flattened tensor)
    input_ptr,  # Input pointer (flattened tensor)
    d: int,  # Dimension size
    BLOCK_SIZE: tl.constexpr,  # Block size (threads per block)
    ACTIVATION_TYPE: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    # Program IDs
    # Cast to int64 to prevent from Invalid memory access:
    # https://github.com/triton-lang/triton/issues/1058
    token_id = tl.program_id(0).to(
        tl.int64
        if USE_INT64 else tl.int32)  # Get token index for this kernel instance
    block_start = tl.arange(
        0, BLOCK_SIZE)  # Create a range of BLOCK_SIZE elements

    # Precompute base pointers for efficiency
    input_base_ptr = input_ptr + token_id * d
    out_ptr_base = out_ptr + token_id * d

    # Iterate over all blocks in the dimension to process all elements
    for i in range(0, d, BLOCK_SIZE):
        block_id = block_start + i

        # Mask to ensure we do not read/write out of bounds
        mask = block_id < d

        # Calculate pointers for the input and output
        input_ptr_curr = input_base_ptr + block_id  # Input values

        # Load input values with masking
        input_vals = tl.load(input_ptr_curr, mask=mask)  # Load input values

        # Apply the selected activation function on input_vals
        if ACTIVATION_TYPE == GELU_NEW_FUNC_TYPE:
            activated_vals = _gelu_new_kernel(input_vals)
        elif ACTIVATION_TYPE == GELU_FAST_FUNC_TYPE:
            activated_vals = _gelu_fast_kernel(input_vals)
        elif ACTIVATION_TYPE == GELU_QUICK_FUNC_TYPE:
            activated_vals = _gelu_quick_kernel(input_vals)
        else:
            activated_vals = input_vals  # No activation if the type is unknown

        # Store the result in the output tensor
        tl.store(out_ptr_base + block_id, activated_vals, mask=mask)


def launch_activation_kernel(out: torch.Tensor, input: torch.Tensor,
                             activation_type: tl.constexpr):
    """
    Launches the Triton activation kernel with the specified 
    activation function.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., d].
    - activation_type (tl.constexpr): Type of activation to be applied.
    """
    # Assuming input has shape [..., d] and out has the same shape
    d: int = input.shape[-1]  # The dimension of the output
    num_tokens: int = input.numel() // input.shape[-1]  # Number of tokens

    # Define block size (maximum threads per block is typically 1024 in Triton)
    BLOCK_SIZE: int = min(triton.next_power_of_2(d), 1024)

    # Check if we need to use int64 for indexing.
    use_int64 = input.numel() > 2**31 - 1

    # Launch the Triton kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        activation_kernel[(num_tokens, )](
            out,
            input,
            d,
            BLOCK_SIZE=BLOCK_SIZE,
            ACTIVATION_TYPE=activation_type,
            USE_INT64=use_int64,
        )


@torch.library.custom_op("triton::gelu_new",
                         mutates_args=["out"],
                         device_types="cuda")
def gelu_new(out: torch.Tensor, input: torch.Tensor) -> None:
    """
    Applies the GeLU activation function with 'new' approximation to `input`
    and stores the result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., d].
    """
    launch_activation_kernel(out, input, GELU_NEW_FUNC_TYPE)


@torch.library.custom_op("triton::gelu_fast",
                         mutates_args=["out"],
                         device_types="cuda")
def gelu_fast(out: torch.Tensor, input: torch.Tensor) -> None:
    """
    Applies the GeLU activation function with 'fast' approximation to `input`
    and stores the result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., d].
    """
    launch_activation_kernel(out, input, GELU_FAST_FUNC_TYPE)


@torch.library.custom_op("triton::gelu_quick",
                         mutates_args=["out"],
                         device_types="cuda")
def gelu_quick(out: torch.Tensor, input: torch.Tensor) -> None:
    """
    Applies the GeLU activation function with 'quick' approximation to `input`
    and stores the result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., d].
    """
    launch_activation_kernel(out, input, GELU_QUICK_FUNC_TYPE)
