from typing import Optional, Tuple
import triton
import triton.language as tl
import torch

from vllm.triton_utils import libentry

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
    inner = tl.cast(0.79788456, tl.float32) * (x + (tl.cast(0.044715, tl.float32) * x_cube_f).to(x.dtype)).to(tl.float32)
    t = _tanhf(inner.to(x.dtype)).to(x.dtype)
    return tl.cast(0.5, x.dtype) * x * (tl.cast(1.0, x.dtype) + t)

@triton.jit
def _gelu_fast_kernel(x):
    # GELU kernel based on the 'fast' approximation
    # Similar to the CUDA code for fast approximation
    x_f = tl.cast(x, tl.float32)
    t = _tanhf((tl.cast(0.79788456, tl.float32) * x_f).to(x.dtype) * (tl.cast(1.0, x.dtype) + (tl.cast(0.044715, tl.float32) * x_f).to(x.dtype) * x))
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


@libentry()
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
    # Cast to int64 to prevent from Invalid memory access: https://github.com/triton-lang/triton/issues/1058
    token_id = tl.program_id(0).to(tl.int64 if USE_INT64 else tl.int32)  # Get token index for this kernel instance
    block_start = tl.arange(0, BLOCK_SIZE)  # Create a range of BLOCK_SIZE elements

    # Precompute common base pointers outside the loop for efficiency
    input_base_ptr1 = input_ptr + token_id * 2 * d
    input_base_ptr2 = input_ptr + token_id * 2 * d + d
    out_ptr_base = out_ptr + token_id * d

    # Iterate over all blocks in the dimension to process all elements
    for i in range(0, d, BLOCK_SIZE):
        block_id = block_start + i

        # Mask to ensure we do not read/write out of bounds
        mask = block_id < d

        # Calculate pointers for the two parts of the input: input and gating values
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

def launch_activation_and_mul(out: torch.Tensor, input: torch.Tensor, activation_type: int) -> None:
    """
    Applies activation function to the first half of `input` and
    multiplies it element-wise with the second half of `input`, storing the
    result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., 2 * d].
    """
    # Assuming input has shape [..., 2 * d] and out has shape [..., d]
    d: int = input.shape[-1] // 2  # The dimension of the output (half of input's last dim)
    num_tokens: int = input.numel() // input.shape[-1]  # Number of tokens

    # Define block size (maximum threads per block is typically 1024 in Triton)
    BLOCK_SIZE: int = min(triton.next_power_of_2(d), 1024)

    # Check if we need to use int64 for indexing.
    use_int64 = input.numel() > 2**31 - 1

    # Launch the Triton kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        act_and_mul_kernel[(num_tokens,)](
            out, input, 
            d,
            BLOCK_SIZE=BLOCK_SIZE,
            ACTIVATION_TYPE=activation_type,
            USE_INT64=use_int64,
        )

@torch.library.custom_op("triton::silu_and_mul",
                         mutates_args=["out"], device_types="cuda")
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
                         mutates_args=["out"], device_types="cuda")
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
                         mutates_args=["out"], device_types="cuda")
def gelu_tanh_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    """
    Applies the GeLU activation function with 'tanh' approximation to the first half of `input`
    and multiplies it element-wise with the second half of `input`, storing the result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., 2 * d].
    """
    launch_activation_and_mul(out, input, GELU_TANH_FUNC_TYPE)


@libentry()
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
    # Cast to int64 to prevent from Invalid memory access: https://github.com/triton-lang/triton/issues/1058
    token_id = tl.program_id(0).to(tl.int64 if USE_INT64 else tl.int32)  # Get token index for this kernel instance
    block_start = tl.arange(0, BLOCK_SIZE)  # Create a range of BLOCK_SIZE elements

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


def launch_activation_kernel(out: torch.Tensor, input: torch.Tensor, activation_type: tl.constexpr):
    """
    Launches the Triton activation kernel with the specified activation function.

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
        activation_kernel[(num_tokens,)](
            out, input, d,
            BLOCK_SIZE=BLOCK_SIZE,
            ACTIVATION_TYPE=activation_type,
            USE_INT64=use_int64,
        )

@torch.library.custom_op("triton::gelu_new",
                         mutates_args=["out"], device_types="cuda")
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
                         mutates_args=["out"], device_types="cuda")
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
                         mutates_args=["out"], device_types="cuda")
def gelu_quick(out: torch.Tensor, input: torch.Tensor) -> None:
    """
    Applies the GeLU activation function with 'quick' approximation to `input`
    and stores the result in `out`.

    Parameters:
    - out (torch.Tensor): Output tensor to store the result with shape [..., d].
    - input (torch.Tensor): Input tensor with shape [..., d].
    """
    launch_activation_kernel(out, input, GELU_QUICK_FUNC_TYPE)


@libentry()
@triton.jit(do_not_specialize=["epsilon"])
def rms_norm_kernel(
    out_ptr, 
    input_ptr, 
    weight_ptr, 
    hidden_size,
    epsilon, 
    BLOCK_SIZE: tl.constexpr
):
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
        input_val = tl.load(input_ptr + block_id * hidden_size + current_idx, mask=mask).to(tl.float32)

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
        input_val = tl.load(input_ptr + block_id * hidden_size + current_idx, mask=mask)
        weight = tl.load(weight_ptr + current_idx, mask=mask)

        # Normalize input and apply weight
        normalized_input = (input_val.to(tl.float32) * norm_factor).to(input_val.dtype) * weight

        # Store the result
        tl.store(out_ptr + block_id * hidden_size + current_idx, normalized_input, mask=mask)


@torch.library.custom_op("triton::rms_norm", mutates_args=["out"], device_types="cuda")
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None:
    # Extract shapes
    num_tokens, hidden_size = input.shape

    # Define the grid/block size for Triton
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 1024)  # A fixed block size for processing chunks of hidden_size
    grid = (num_tokens,)

    # Launch the kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        rms_norm_kernel[grid](out, input, weight, hidden_size, epsilon, BLOCK_SIZE)


@libentry()
@triton.jit(do_not_specialize=["epsilon"])
def fused_add_rms_norm_kernel(
    input_ptr, residual_ptr, weight_ptr, 
    hidden_size, epsilon, BLOCK_SIZE: tl.constexpr
):
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
        input_val = tl.load(input_ptr + block_id * hidden_size + current_idx, mask=mask)
        residual_val = tl.load(residual_ptr + block_id * hidden_size + current_idx, mask=mask)
        fused_val = input_val + residual_val
        tl.store(residual_ptr + block_id * hidden_size + current_idx, fused_val, mask=mask)

        # Accumulate sum of squares for variance calculation
        fused_val = tl.cast(fused_val, tl.float32)
        sum_squares += tl.sum(tl.where(mask, fused_val * fused_val, 0.0), axis=0)

    # Compute variance and RMS normalization factor
    variance = sum_squares / hidden_size
    norm_factor = tl.rsqrt(variance + epsilon)

    # Second pass: Apply normalization and weight
    for offset in range(0, hidden_size, BLOCK_SIZE):
        current_idx = idx + offset
        mask = current_idx < hidden_size

        # Load fused values and weight, normalize and scale
        x = tl.load(residual_ptr + block_id * hidden_size + current_idx, mask=mask)
        weight = tl.load(weight_ptr + current_idx, mask=mask)
        normalized_input = (x.to(tl.float32) * norm_factor).to(x.dtype) * weight

        # Store the result in the output tensor
        tl.store(input_ptr + block_id * hidden_size + current_idx, normalized_input, mask=mask)

# Triton custom op for fused_add_rms_norm
@torch.library.custom_op("triton::fused_add_rms_norm", mutates_args=["input", "residual"], device_types="cuda")
def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None:
    # Get dimensions
    num_tokens, hidden_size = input.shape

    # Define grid and block size
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 1024)
    grid = (num_tokens,)

    # Launch the kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        fused_add_rms_norm_kernel[grid](
            input, residual, weight, hidden_size, epsilon, BLOCK_SIZE
        )


@triton.jit
def _apply_rotary_embedding_kernel(
    query_ptr,         # [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size] 
    key_ptr,          # [batch_size, seq_len, num_kv_heads * head_size] or [num_tokens, num_kv_heads * head_size]
    cos_sin_cache_ptr, # [max_position, rot_dim]
    rot_dim,          # rotation dimension
    query_token_offset,     
    key_token_offset,       
    num_heads,        # number of attention heads
    num_kv_heads,     # number of key/value heads
    head_size,        # size of each head
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


@libentry()
@triton.jit
def rotary_embedding_kernel(
    positions_ptr,      # [batch_size, seq_len] or [num_tokens]
    query_ptr,         # [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size] 
    key_ptr,          # [batch_size, seq_len, num_kv_heads * head_size] or [num_tokens, num_kv_heads * head_size]
    cos_sin_cache_ptr, # [max_position, rot_dim]
    num_tokens,        # number of total tokens
    rot_dim,          # rotation dimension
    query_stride,     # stride for query tensor
    key_stride,       # stride for key tensor
    num_heads,        # number of attention heads
    num_kv_heads,     # number of key/value heads
    head_size,        # size of each head
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
    
    _apply_rotary_embedding_kernel(
        query_ptr, 
        key_ptr, 
        cache_ptr, 
        rot_dim, 
        query_token_offset, 
        key_token_offset, 
        num_heads, 
        num_kv_heads, 
        head_size, 
        is_neox, 
        BLOCK_SIZE 
    )


@torch.library.custom_op("triton::rotary_embedding", mutates_args=["query", "key"], device_types="cuda")
def rotary_embedding(
    positions: torch.Tensor,  # [batch_size, seq_len] or [num_tokens]
    query: torch.Tensor,      # [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    key: torch.Tensor,        # [batch_size, seq_len, num_kv_heads * head_size] or [num_tokens, num_kv_heads * head_size]
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
    grid = (num_tokens,)
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


@libentry()
@triton.jit
def batched_rotary_embedding_kernel(
    # Pointers to matrices
    positions_ptr,         # [num_tokens]
    query_ptr,            # [num_tokens, num_heads, head_size]
    key_ptr,             # [num_tokens, num_kv_heads, head_size]
    cos_sin_cache_ptr,    # [max_position, 2, rot_dim // 2]
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
    
    _apply_rotary_embedding_kernel(
        query_ptr, 
        key_ptr, 
        cache_ptr, 
        rot_dim, 
        query_token_offset, 
        key_token_offset, 
        num_heads, 
        num_kv_heads, 
        head_size, 
        is_neox, 
        BLOCK_SIZE 
    )

@torch.library.custom_op("triton::batched_rotary_embedding", mutates_args=["query", "key"], device_types="cuda")
def batched_rotary_embedding(
    positions: torch.Tensor,          # [num_tokens]
    query: torch.Tensor,              # [num_tokens, num_heads * head_size]
    key: torch.Tensor,                # [num_tokens, num_kv_heads * head_size]
    head_size: int,
    cos_sin_cache: torch.Tensor,      # [max_position, rot_dim]
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
    grid = (num_tokens,)
    
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


@triton.jit
def _float_to_int8_rn(x: tl.tensor) -> tl.tensor:
    """
    Convert float to int8 with round-to-nearest-even behavior using CUDA libdevice.
    
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
    Convert float to int32 with round-to-nearest-even behavior using CUDA libdevice.
    
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

@libentry()
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

@torch.library.custom_op("triton::static_scaled_int8_quant", mutates_args=["out"], device_types="cuda")
def static_scaled_int8_quant(
    out: torch.Tensor,
    input: torch.Tensor, 
    scale: torch.Tensor,
    azp: Optional[torch.Tensor] = None
) -> None:
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
    grid = (num_tokens,)
    
    # Launch kernel
    static_scaled_int8_quant_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        scale_ptr=scale,
        azp_ptr=azp,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

@libentry()
@triton.jit
def dynamic_scaled_int8_quant_kernel(
    input_ptr,    # 输入tensor指针 (fp16/bf16/fp32)
    output_ptr,   # 输出tensor指针 (int8)
    scale_ptr,    # scale tensor指针 (fp32) 
    hidden_size,  # hidden dimension大小
    BLOCK_SIZE: tl.constexpr,  # triton block size
):
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前token的offset
    token_offset = pid * hidden_size + tl.arange(0, BLOCK_SIZE)
    max_token_offset = (pid + 1) * hidden_size
    
    # 加载数据并计算最大绝对值
    max_val = tl.cast(float('-inf'), tl.float32)
    for start_idx in range(0, hidden_size, BLOCK_SIZE):
        # 加载输入数据
        offset = token_offset + start_idx
        mask = offset < max_token_offset
        x = tl.load(input_ptr + offset, mask=mask, other=0.0)
        
        # 计算绝对值并更新最大值
        x_abs = tl.abs(x) 
        max_val = tl.maximum(max_val, tl.max(x_abs, axis=0).to(tl.float32))
    
    # 计算scale并存储
    scale = max_val / 127.0
    tl.store(scale_ptr + pid, scale)
    
    # 量化数据
    scale_recip = 127.0 / max_val
    for start_idx in range(0, hidden_size, BLOCK_SIZE):
        offset = token_offset + start_idx
        mask = offset < max_token_offset
        x = tl.load(input_ptr + offset, mask=mask)

        # Convert to float32 for computation
        x = x.to(tl.float32)
        
        # 量化到int8
        x_scaled = x * scale_recip
        x_int8 = _float_to_int8_rn(x_scaled)
        
        tl.store(output_ptr + offset, x_int8, mask=mask)

@libentry()
@triton.jit
def dynamic_scaled_int8_azp_quant_kernel(
    input_ptr,    # 输入tensor指针
    output_ptr,   # 输出tensor指针  
    scale_ptr,    # scale tensor指针
    azp_ptr,      # zero point tensor指针
    hidden_size,  # hidden dimension大小
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offset = pid * hidden_size + tl.arange(0, BLOCK_SIZE)
    max_token_offset = (pid + 1) * hidden_size
    
    # 第一遍:找到最大最小值
    min_val = tl.cast(float('inf'), tl.float32)
    max_val = tl.cast(float('-inf'), tl.float32)
    
    for start_idx in range(0, hidden_size, BLOCK_SIZE):
        offset = token_offset + start_idx
        mask = offset < max_token_offset
        x = tl.load(input_ptr + offset, mask=mask)
        
        min_val = tl.minimum(min_val, tl.min(tl.where(mask, x, tl.cast(float('inf'), x.dtype)), axis=0).to(tl.float32))
        max_val = tl.maximum(max_val, tl.max(tl.where(mask, x, tl.cast(float('-inf'), x.dtype)), axis=0).to(tl.float32))
    
    # 计算scale和zero point
    scale = (max_val - min_val) / 255.0
    azp = tl.extra.cuda.libdevice.nearbyint(-128.0 - min_val / scale)
    
    # 存储scale和zero point
    tl.store(scale_ptr + pid, scale)
    tl.store(azp_ptr + pid, tl.cast(azp, tl.int32))
    
    # 第二遍:量化数据
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

@torch.library.custom_op("triton::dynamic_scaled_int8_quant", mutates_args=["out", "scales", "azp"], device_types="cuda")
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

    grid = (num_tokens,)
    # 根据是否有azp选择kernel
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
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
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
    dynamic_scaled_int8_quant(output, input, input_scales,
                                           input_azp)
    return output, input_scales, input_azp


def get_scaled_mm_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

@libentry()
@triton.autotune(
    configs=get_scaled_mm_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def scaled_mm_kernel(
    # Output pointer
    c_ptr, 
    # Input pointers
    a_ptr, b_ptr, 
    # Pointers to scales and zero points
    a_scales_ptr, b_scales_ptr, 
    azp_adj_ptr, azp_ptr,
    # Bias pointer (optional)
    bias_ptr,
    # Matrix dimensions 
    M, N, K,
    # Whether to use bias
    HAS_BIAS: tl.constexpr,
    # Stride information
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
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
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of int32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        
        # Accumulate matrix multiplication in int32
        accumulator += tl.dot(a, b, out_dtype=tl.int32)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Handle AZP correction
    if HAS_AZP_ADJ:
        if HAS_AZP:
            # Per-token AZP case
            azp = tl.load(azp_ptr + offs_am).to(dtype=tl.int32)  # Shape: [BLOCK_SIZE_M] dtype=int32
            azp_adj = tl.load(azp_adj_ptr + offs_bn).to(dtype=tl.int32)  # Shape: [BLOCK_SIZE_N] dtype=int32
            azp_correction = azp[:, None] * azp_adj[None, :]
        else:
            # Per-tensor AZP case
            azp_correction = tl.load(azp_adj_ptr + offs_bn)  # Shape: [BLOCK_SIZE_N]
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
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    accumulator = accumulator.to(dtype=OUT_DTYPE)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# TODO support FP8
@torch.library.custom_op("triton::scaled_mm", mutates_args=["out"], device_types="cuda")
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
    M, K = a.shape
    _, N = b.shape

    # Calculate grid size to cover entire output matrix
    # Each block computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of the output
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # Determine if we have per-row/per-column scaling
    # a_scale_rows: True if we have different scale per row of A
    # b_scale_cols: True if we have different scale per column of B
    a_scale_rows = (a_scales.numel() == M)
    b_scale_cols = (b_scales.numel() == N)
    
    # Launch the CUDA kernel on the appropriate device
    device = torch.cuda.device_of(a)
    with torch.cuda.device(device):
        scaled_mm_kernel[grid](
            out, a, b, a_scales, b_scales,
            azp_adj if azp_adj is not None else a,  # Use a as placeholder if no azp_adj
            azp if azp is not None else a,  # Use a as placeholder if no azp
            bias if bias is not None else a,  # Use a as placeholder if no bias
            M, N, K,
            bias is not None,  # HAS_BIAS flag
            a.stride(0), a.stride(1),  # Strides for matrix A
            b.stride(0), b.stride(1),  # Strides for matrix B
            out.stride(0), out.stride(1),  # Strides for output matrix
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
    Performs quantized matrix multiplication: C = (a_scales * A) @ (b_scales * B) + bias
    
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


def triton_scaled_mm_azp(a: torch.Tensor, b: torch.Tensor,
                        a_scales: torch.Tensor, b_scales: torch.Tensor,
                        out_dtype: torch.dtype,
                        azp_adj: torch.Tensor,
                        azp: Optional[torch.Tensor] = None,
                        bias: Optional[torch.Tensor] = None):
    """
    Wrapper function for the Triton kernel implementing scaled matrix multiplication with AZP.
    
    Computes: C = (a_scales * (A - azp)) @ (b_scales * B) + bias
    where azp_adj = sum(B, dim=0) is precomputed for efficiency
    """
    # Step 1: Validate input tensor data types
    # A and B must be int8 or float8, scales must be float32
    assert a.dtype in [torch.int8, torch.float8_e4m3fn]
    assert b.dtype == a.dtype
    assert a_scales.dtype == torch.float32
    assert b_scales.dtype == torch.float32

    # Step 3: Extract matrix dimensions
    # M = batch size, K = hidden dim, N = output dim
    M, K = a.shape
    _, N = b.shape
    
    # Step 4: Allocate output tensor with specified dtype
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    
    # Step 5: Call Triton kernel for scaled matrix multiply with AZP
    scaled_mm(c, a, b, a_scales, b_scales, azp_adj, azp, bias)
    
    # Step 6: Return computed result
    return c
