import torch
import triton
import triton.testing

from vllm import _custom_ops as cuda_ops
from vllm import _triton_ops as triton_ops
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser,
                        seed_everything)


def create_benchmark_configs():
    configs = []
    for hidden_size in [4096, 8192, 12288, 16384]:
        for static_scale in [True, False]:
            for dtype in ["half", "bfloat16", "float"]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["num_tokens"],  # Variable parameter
                        x_vals=[1024, 2048, 4096, 8192],  # Parameter values
                        line_arg="version",  # Argument to compare
                        line_vals=["cuda", "triton"],  # Values to compare
                        line_names=["CUDA", "Triton"],  # Legend names
                        styles=[("blue", "-"), ("red", "-")],  # Line styles
                        ylabel="Runtime (ms)",  # Y-axis label
                        plot_name=
                        f"quant-hidden_size-{hidden_size}-dtype-{dtype}-static_scale-{static_scale}",
                        args={
                            "hidden_size": hidden_size,
                            "dtype": STR_DTYPE_TO_TORCH_DTYPE[dtype],
                            "static_scale": static_scale,
                        },
                    ))
    return configs


@triton.testing.perf_report(create_benchmark_configs())
def benchmark_quant(
    num_tokens: int,
    hidden_size: int,
    quant_type: str,
    dtype: torch.dtype,
    static_scale: bool = True,
    seed: int = 0,
    version: str = "cuda",
    device: str = "cuda",
) -> float:
    seed_everything(seed)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    scale = torch.randn(1, 1, dtype=torch.float32,
                        device=device) if static_scale else None
    quant_dtype = torch.int8 if quant_type == "int8" else torch.float8_e4m3fn

    ops = cuda_ops if version == "cuda" else triton_ops

    def run_kernel():
        if quant_dtype == torch.int8:
            ops.scaled_int8_quant(x, scale)
        else:
            ops.scaled_fp8_quant(x, scale)

    # Use triton's timer to measure performance
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_kernel(),
                                                 quantiles=quantiles)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark the quantization (fp8 or int8) kernel.")
    parser.add_argument("--quant-type",
                        type=str,
                        choices=["fp8", "int8"],
                        default="int8")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    print(args)

    benchmark_quant.run(print_data=True,
                        save_path="benchmark_quant_res",
                        quant_type=args.quant_type,
                        seed=args.seed)
