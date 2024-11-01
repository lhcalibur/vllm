import argparse
import copy
import itertools
import pickle as pkl
import time
from typing import Iterable, List, Optional, Tuple

import torch
import triton
import triton.testing
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as cuda_ops
from vllm import _triton_ops as triton_ops
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]


# helpers
def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    if dtype == torch.int8:
        return to_int8(a), to_int8(b)
    if dtype == torch.float8_e4m3fn:
        return to_fp8(a), to_fp8(b)

    raise ValueError("unsupported dtype")


# bench
def bench_int8(
        dtype: torch.dtype,
        MKNs: Iterable[Tuple[int, int, int]],
        tp_size: Optional[int] = None) -> List[triton.testing.Benchmark]:
    assert dtype == torch.int8
    configs = []
    for with_bias in [True, False]:
        for with_azp_per_tensor in [True, False]:
            for with_azp_per_token in [True, False]:
                plot_name = (f"int8_gemm-{dtype}-bias-{with_bias}-"
                             f"azp-per-tensor-{with_azp_per_tensor}-"
                             f"azp-per-token-{with_azp_per_token}")
                if tp_size is not None:
                    plot_name += f"-tp-{tp_size}"
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["m", "k", "n"],
                        x_vals=MKNs,
                        line_arg="version",
                        line_vals=["cuda", "triton"],
                        line_names=["CUDA", "Triton"],
                        styles=[("blue", "-"), ("red", "-")],
                        ylabel="Runtime (ms)",
                        plot_name=plot_name,
                        args={
                            "dtype": dtype,
                            "with_bias": with_bias,
                            "with_azp_per_tensor": with_azp_per_tensor,
                            "with_azp_per_token": with_azp_per_token,
                        },
                    ))

    def _bench(dtype: torch.dtype, m: int, k: int, n: int, with_bias: bool,
               with_azp_per_tensor: bool, with_azp_per_token: bool,
               version: str):
        a, b = make_rand_tensors(dtype, m, n, k)
        scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)
        azp = torch.zeros((m, ), device="cuda", dtype=torch.int32)
        azp_adj = torch.zeros((n, ), device="cuda", dtype=torch.int32)

        args = [a, b, scale_a, scale_b, torch.bfloat16]
        kwargs = {}
        if with_azp_per_token:
            kwargs["azp_adj"] = azp_adj
            kwargs["azp"] = azp
        if with_azp_per_tensor:
            kwargs["azp_adj"] = azp_adj
            kwargs["azp"] = None
        if with_bias:
            kwargs["bias"] = bias

        if version == "cuda":
            if with_azp_per_tensor or with_azp_per_token:
                ops = cuda_ops.cutlass_scaled_mm_azp
            else:
                ops = cuda_ops.cutlass_scaled_mm
        if version == "triton":
            if with_azp_per_tensor or with_azp_per_token:
                ops = triton_ops.triton_scaled_mm_azp
            else:
                ops = triton_ops.triton_scaled_mm

        def bench_fn():
            ops(*args, **kwargs)

        ms, min_ms, max_ms = triton.testing.do_bench(bench_fn,
                                                     quantiles=[0.5, 0.2, 0.8])
        return ms, max_ms, min_ms

    triton.testing.perf_report(configs)(_bench).run()


# TODO support fp8
def bench(
        dtype: torch.dtype,
        MKNs: Iterable[Tuple[int, int,
                             int]]) -> List[triton.testing.Benchmark]:
    if dtype == torch.int8:
        return bench_int8(dtype, MKNs)
    if dtype == torch.float8_e4m3fn:
        # return bench_fp8(dtype, m, k, n)
        raise NotImplementedError("FP8 is not supported yet")
    raise ValueError("unsupported type")


def run(dtype: torch.dtype, MKNs: Iterable[Tuple[int, int, int]]):
    bench(dtype, MKNs)


def make_output(data,
                MKNs: Iterable[Tuple[int, int, int]],
                base_description: str,
                timestamp=None):
    print(f"== All Results {base_description} ====")
    for result in data:
        print(result)

    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


def run_square_bench(args):
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    run(args.dtype, MKNs)


def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    run(args.dtype, MKNs)


def run_model_bench(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
        KNs = []
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KNs.append(KN)
        return KNs

    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))

        run(args.dtype, MKNs)


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError("unsupported dtype")

    parser = FlexibleArgumentParser(
        description="""
Benchmark Cutlass GEMM.

    To run square GEMMs:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \\
            --dtype fp8 square_bench --dim-start 128 \\
            --dim-end 512 --dim-increment 64
    
    To run constant N and K and sweep M:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \\
            --dtype fp8 range_bench --dim-start 128 --dim-end 512 \\
            --dim-increment 64 --n-constant 16384 --k-constant 16384
    
    To run dimensions from a model:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py \\
            --dtype fp8 model_bench --models meta-llama/Llama-2-7b-hf \\
            --batch-sizes 16 --tp-sizes 1
    
    Output:
        - a .pkl file containing raw torch.benchmark.utils.Measurements \\
          for the pytorch and cutlass implementations for the various GEMMs.
            """,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--dtype",
                        type=to_torch_dtype,
                        required=True,
                        help="Available options are ['int8', 'fp8']")
    subparsers = parser.add_subparsers(dest="cmd")

    square_parser = subparsers.add_parser("square_bench")
    square_parser.add_argument("--dim-start", type=int, required=True)
    square_parser.add_argument("--dim-end", type=int, required=True)
    square_parser.add_argument("--dim-increment", type=int, required=True)
    square_parser.set_defaults(func=run_square_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--dim-start", type=int, required=True)
    range_parser.add_argument("--dim-end", type=int, required=True)
    range_parser.add_argument("--dim-increment", type=int, required=True)
    range_parser.add_argument("--m-constant", type=int, default=None)
    range_parser.add_argument("--n-constant", type=int, default=None)
    range_parser.add_argument("--k-constant", type=int, default=None)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument("--models",
                              nargs="+",
                              type=str,
                              default=DEFAULT_MODELS,
                              choices=WEIGHT_SHAPES.keys())
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--batch-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_BATCH_SIZES)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)
