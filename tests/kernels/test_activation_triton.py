from typing import Type

import pytest
import torch

from vllm import _triton_ops as ops
from vllm.model_executor.layers.activation import (FastGELU, GeluAndMul,
                                                   NewGELU, QuickGELU,
                                                   SiluAndMul)
from vllm.utils import seed_everything

from .allclose_default import get_default_atol, get_default_rtol

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048, 131072]  # Arbitrary values for testing
D = [512, 4096, 5120, 13824, 14336]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu":
        layer = SiluAndMul()
    elif activation == "gelu":
        layer = GeluAndMul(approximate="none")
    elif activation == "gelu_tanh":
        layer = GeluAndMul(approximate="tanh")
    out = layer.forward_triton(x)

    # Prevent assert_close from OOM
    if num_tokens > 13824 or d > 8192:
        out = out.cpu()
        ref_out = layer.forward_cuda(x).cpu()
    else:
        ref_out = layer.forward_native(x)

    torch.testing.assert_close(out,
                               ref_out,
                               atol=get_default_atol(out),
                               rtol=get_default_rtol(out))


@pytest.mark.parametrize("activation", [(FastGELU, ops.gelu_fast),
                                        (NewGELU, ops.gelu_new),
                                        (QuickGELU, ops.gelu_quick)])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_activation(
    activation: Type[torch.nn.Module],
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = activation[0]()
    out = layer.forward_triton(x)
    ref_out = layer.forward_native(x)
    torch.testing.assert_close(out,
                               ref_out,
                               atol=get_default_atol(out),
                               rtol=get_default_rtol(out))
