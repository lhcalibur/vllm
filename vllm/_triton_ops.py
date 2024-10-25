import triton
import triton.language as tl
import torch
from enum import Enum


SILU_FUNC_TYPE: tl.constexpr = 0
GELU_FUNC_TYPE: tl.constexpr = 1
GELU_TANH_FUNC_TYPE: tl.constexpr = 2
GELU_NEW_FUNC_TYPE: tl.constexpr = 3
GELU_FAST_FUNC_TYPE: tl.constexpr = 4
GELU_QUICK_FUNC_TYPE: tl.constexpr = 5

@triton.jit
def silu_kernel(x):
    # SiLU (x * sigmoid(x))

    # # Use 2^x instead of e^x results in an additional scale factor of log2(e)
    # return x / (1.0 + tl.exp2(-x * 1.44269504089))

    x = tl.cast(x, tl.float32)
    return x * tl.sigmoid(x)

@triton.jit
def gelu_kernel(x):
    # GELU kernel based on CUAD code
    # Equivalent to PyTorch GELU with 'none' approximation
    M_SQRT1_2 = 0.70710678118654752440  # 1/sqrt(2)
    return x * 0.5 * (1.0 + tl.erf(x * M_SQRT1_2))

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu_tanh_kernel(x):
    # GELU kernel based on 'tanh' approximation
    # Equivalent to PyTorch GELU with 'tanh' approximation
    M_SQRT2 = 1.41421356237309504880  # sqrt(2)
    M_2_SQRTPI = 1.12837916709551257390  # 2/sqrt(pi)
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715

    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return x * 0.5 * (1.0 + tanh(inner))

@triton.jit
def gelu_new_kernel(x):
    # GELU kernel based on the 'new' approximation
    # Equivalent to the CUDA code provided
    x_cube = x * x * x
    inner = 0.79788456 * (x + 0.044715 * x_cube)
    t = tanh(inner)
    return 0.5 * x * (1.0 + t)

@triton.jit
def gelu_fast_kernel(x):
    # GELU kernel based on the 'fast' approximation
    # Similar to the CUDA code for fast approximation
    f = x
    t = tanh(0.79788456 * f * (1.0 + 0.044715 * f * x))
    return 0.5 * x * (1.0 + t)

@triton.jit
def gelu_quick_kernel(x):
    # GELU kernel based on the 'quick' approximation
    # Equivalent to x * sigmoid(1.702 * x)
    # Use 2^x instead of e^x results in an additional scale factor of log2(e)
    # return x / (1.0 + tl.exp2(-1.702 * x * 1.44269504089))

    x = tl.cast(x, tl.float32)
    return x * tl.sigmoid(1.702 * x)


@triton.jit
def act_and_mul_kernel(
    out_ptr,  # Output pointer (flattened tensor)
    input_ptr,  # Input pointer (flattened tensor)
    D: tl.constexpr,  # Dimension size
    BLOCK_SIZE: tl.constexpr,  # Block size (threads per block)
    ACTIVATION_TYPE: tl.constexpr, 
):
    # Program IDs
    token_id = tl.program_id(0)  # Get token index for this kernel instance
    block_start = tl.arange(0, BLOCK_SIZE)  # Create a range of BLOCK_SIZE elements

    # Precompute common base pointers outside the loop for efficiency
    input_base_ptr1 = input_ptr + token_id * 2 * D
    input_base_ptr2 = input_ptr + token_id * 2 * D + D
    out_ptr_base = out_ptr + token_id * D

    # Iterate over all blocks in the dimension to process all elements
    for i in range(0, D, BLOCK_SIZE):
        block_id = block_start + i

        # Mask to ensure we do not read/write out of bounds
        mask = block_id < D

        # Calculate pointers for the two parts of the input: input and gating values
        input_ptr1 = input_base_ptr1 + block_id  # First half (input values)
        input_ptr2 = input_base_ptr2 + block_id  # Second half (gate values)

        # Load input and gate values with masking
        input_vals = tl.load(input_ptr1, mask=mask, other=0.0)  # Load input values
        gate_vals = tl.load(input_ptr2, mask=mask, other=0.0)  # Load gate values

        # Apply the selected activation function on input_vals
        if ACTIVATION_TYPE == SILU_FUNC_TYPE:
            activated_vals = silu_kernel(input_vals)
        elif ACTIVATION_TYPE == GELU_FUNC_TYPE:
            activated_vals = gelu_kernel(input_vals)
        elif ACTIVATION_TYPE == GELU_TANH_FUNC_TYPE:
            activated_vals = gelu_tanh_kernel(input_vals)
        else:
            activated_vals = input_vals  # No activation if the type is unknown

        # Elementwise multiply activation with gate_vals
        result = activated_vals * gate_vals

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
    BLOCK_SIZE: int = min(d, 1024)

    # Launch the Triton kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        act_and_mul_kernel[(num_tokens,)](
            out, input, d,
            BLOCK_SIZE=BLOCK_SIZE,
            ACTIVATION_TYPE=activation_type,
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


@triton.jit
def activation_kernel(
    out_ptr,  # Output pointer (flattened tensor)
    input_ptr,  # Input pointer (flattened tensor)
    D: tl.constexpr,  # Dimension size
    BLOCK_SIZE: tl.constexpr,  # Block size (threads per block)
    ACTIVATION_TYPE: tl.constexpr,
):
    # Program IDs
    token_id = tl.program_id(0)  # Get token index for this kernel instance
    block_start = tl.arange(0, BLOCK_SIZE)  # Create a range of BLOCK_SIZE elements

    # Precompute base pointers for efficiency
    input_base_ptr = input_ptr + token_id * D
    out_ptr_base = out_ptr + token_id * D

    # Iterate over all blocks in the dimension to process all elements
    for i in range(0, D, BLOCK_SIZE):
        block_id = block_start + i

        # Mask to ensure we do not read/write out of bounds
        mask = block_id < D

        # Calculate pointers for the input and output
        input_ptr_curr = input_base_ptr + block_id  # Input values

        # Load input values with masking
        input_vals = tl.load(input_ptr_curr, mask=mask, other=0.0)  # Load input values

        # Apply the selected activation function on input_vals
        if ACTIVATION_TYPE == GELU_NEW_FUNC_TYPE:
            activated_vals = gelu_new_kernel(input_vals)
        elif ACTIVATION_TYPE == GELU_FAST_FUNC_TYPE:
            activated_vals = gelu_fast_kernel(input_vals)
        elif ACTIVATION_TYPE == GELU_QUICK_FUNC_TYPE:
            activated_vals = gelu_quick_kernel(input_vals)
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
    BLOCK_SIZE: int = min(d, 1024)

    # Launch the Triton kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        activation_kernel[(num_tokens,)](
            out, input, d,
            BLOCK_SIZE=BLOCK_SIZE,
            ACTIVATION_TYPE=activation_type,
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


@triton.jit(do_not_specialize=["epsilon"])
def rms_norm_kernel(
    out_ptr, 
    input_ptr, 
    weight_ptr, 
    hidden_size,
    epsilon, 
    PADDED_HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Compute block and thread indices
    block_id = tl.program_id(0)  # Each block handles one token
    idx = tl.arange(0, BLOCK_SIZE)  # Per-thread index

    # Start the reduction at zero
    sum_squares = tl.zeros((), dtype=tl.float32)  # Accumulate as a scalar

    # Loop over chunks of the hidden_size
    for offset in range(0, PADDED_HIDDEN_SIZE, BLOCK_SIZE):
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
    for offset in range(0, PADDED_HIDDEN_SIZE, BLOCK_SIZE):
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
    padded_hidden_size = triton.next_power_of_2(hidden_size)
    BLOCK_SIZE = min(padded_hidden_size, 1024)  # A fixed block size for processing chunks of hidden_size
    grid = (num_tokens,)

    # Launch the kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        rms_norm_kernel[grid](out, input, weight, hidden_size, epsilon, padded_hidden_size, BLOCK_SIZE)


@triton.jit(do_not_specialize=["epsilon"])
def fused_add_rms_norm_kernel(
    input_ptr, residual_ptr, weight_ptr, 
    hidden_size, epsilon, PADDED_HIDDEN_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Block and thread indexing
    block_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE)

    # Initialize sum of squares accumulator
    sum_squares = tl.zeros((), dtype=tl.float32)

    # Loop over hidden_size in chunks
    for offset in range(0, PADDED_HIDDEN_SIZE, BLOCK_SIZE):
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

    # Second loop to apply normalization and weight
    for offset in range(0, PADDED_HIDDEN_SIZE, BLOCK_SIZE):
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
    num_tokens, hidden_size = input.shape

    # Define grid and block size
    padded_hidden_size = triton.next_power_of_2(hidden_size)
    BLOCK_SIZE = min(padded_hidden_size, 1024)
    grid = (num_tokens,)

    # Launch the kernel
    device = torch.cuda.device_of(input)
    with torch.cuda.device(device):
        fused_add_rms_norm_kernel[grid](
            input, residual, weight, hidden_size, epsilon, 
            padded_hidden_size, BLOCK_SIZE
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


@triton.jit
def _rotary_embedding_kernel(
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
    _rotary_embedding_kernel[grid](
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
def _batched_rotary_embedding_kernel(
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
    
    _batched_rotary_embedding_kernel[grid](
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