import random
from typing import List, Optional

import torch
import triton
import triton.testing

from vllm import _custom_ops as ops
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser,
                        create_kv_caches_with_random, seed_everything)

NUM_BLOCKS = 1024
PARTITION_SIZE = 512


def create_benchmark_configs():
    configs = []
    for head_size in [64, 80, 96, 112, 120, 128, 192, 256]:
        for block_size in [16, 32]:
            for dtype in ["half", "bfloat16", "float"]:
                plot_name = (f"paged-attention-dtype-{dtype}-"
                             f"head_size-{head_size}-"
                             f"block_size-{block_size}")
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["batch_size"],
                        x_vals=[1, 2, 4, 8, 16, 32, 64, 128],
                        line_arg="version",
                        line_vals=["v1", "v2", "triton"],
                        line_names=[
                            "PagedAttn-v1",
                            "PagedAttn-v2",
                            "PagedAttn-Triton",
                        ],
                        styles=[
                            ("blue", "-"),
                            ("red", "-"),
                            ("green", "-"),
                        ],
                        ylabel="Runtime (ms)",
                        plot_name=plot_name,
                        args={
                            "head_size": head_size,
                            "block_size": block_size,
                            "dtype": STR_DTYPE_TO_TORCH_DTYPE[dtype],
                        },
                    ))
    return configs


@triton.testing.perf_report(create_benchmark_configs())
def benchmark_paged_attention(
    batch_size: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    version: str,
    kv_cache_dtype: Optional[str] = None,
    device: str = "cuda",
) -> float:
    seed_everything(seed)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(batch_size,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
    query.uniform_(-scale, scale)

    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device=device)

    seq_lens = [seq_len for _ in range(batch_size)]
    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int, device=device)

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: List[List[int]] = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)
    block_tables = torch.tensor(block_tables_lst,
                                dtype=torch.int,
                                device=device)

    key_caches, value_caches = create_kv_caches_with_random(NUM_BLOCKS,
                                                            block_size,
                                                            1,
                                                            num_kv_heads,
                                                            head_size,
                                                            kv_cache_dtype,
                                                            dtype,
                                                            device=device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    output = torch.empty_like(query)
    if version == "v2":
        num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
        tmp_output = torch.empty(
            size=(batch_size, num_query_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(batch_size, num_query_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

    k_scale = v_scale = 1.0

    def run_kernel():
        if version == "v1":
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        elif version == "v2":
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        elif version == "triton":
            from vllm.attention.ops.paged_attn_triton import paged_attention
            paged_attention(
                output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_kernel(),
                                                 quantiles=quantiles)
    return ms, max_ms, min_ms


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
    )
    args = parser.parse_args()

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    benchmark_paged_attention.run(
        print_data=True,
        save_path='benchmark_paged_attention_res',
        seq_len=args.seq_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        use_alibi=args.use_alibi,
        seed=args.seed,
        kv_cache_dtype=args.kv_cache_dtype,
    )


if __name__ == "__main__":
    main()
