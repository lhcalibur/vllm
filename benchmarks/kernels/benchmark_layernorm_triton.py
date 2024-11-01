import os
from typing import Tuple

import torch
import triton
import triton.testing

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils import FlexibleArgumentParser, seed_everything


def create_benchmark_configs():

    configs = []
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        for add_residual in [True, False]:
            dtype_str = str(dtype).split('.')[-1]
            if add_residual:
                plot_name = f'rms-norm-performance-residule-{dtype_str}'
            else:
                plot_name = f'rms-norm-performance-{dtype_str}'
            configs.append(
                triton.testing.Benchmark(x_names=['num_tokens'],
                                         x_vals=[512, 1024, 2048, 4096, 8192],
                                         line_arg='provider',
                                         line_vals=['cuda', 'triton'],
                                         line_names=['CUDA', 'Triton'],
                                         styles=[('blue', '-'),
                                                 ('green', '-')],
                                         ylabel='Latency (ms)',
                                         plot_name=plot_name,
                                         args={
                                             "dtype": dtype,
                                             "add_residual": add_residual
                                         }))
    return configs


@triton.testing.perf_report(create_benchmark_configs())
def benchmark_rms_norm(num_tokens: int,
                       hidden_size: int,
                       add_residual: bool,
                       dtype: torch.dtype,
                       provider: str,
                       seed: int = 0) -> Tuple[float, float, float]:
    seed_everything(seed)
    torch.set_default_device("cuda")

    os.environ["VLLM_PREFER_TRITON_OPS"] = "1" if provider == 'triton' else "0"

    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    forward_fn = lambda: layer(x, residual)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(forward_fn,
                                                 quantiles=quantiles)

    return ms, min_ms, max_ms


def main(hidden_size: int, seed: int = 0) -> None:
    benchmark_rms_norm.run(print_data=True,
                           save_path='benchmark_layernorm_res',
                           hidden_size=hidden_size,
                           seed=seed)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark the layernorm kernel.")
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    print(args)

    main(hidden_size=args.hidden_size, seed=args.seed)
