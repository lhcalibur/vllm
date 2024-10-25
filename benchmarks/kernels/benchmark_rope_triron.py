from itertools import accumulate
import os
from typing import List, Optional
from itertools import product
import torch
import triton
import triton.testing

from vllm.model_executor.layers.rotary_embedding import (RotaryEmbedding,
                                                       get_rope, _ROPE_DICT)
from vllm.utils import FlexibleArgumentParser, seed_everything
import vllm.envs as envs


# 配置基准测试参数
configs = [
    triton.testing.Benchmark(
        x_names=['head_size'],  # x轴参数名
        x_vals=[64, 80, 96, 112, 120, 128, 192, 256],  # head_size 的可能值
        line_arg='provider',  # 用于区分不同实现的参数
        line_vals=['batched', 'non-batched', 'batched-triton', 'non-batched-triton'],  # 四种实现方式
        line_names=['Batched RoPE', 'Non-batched RoPE', 'Batched RoPE (Triton)', 'Non-batched RoPE (Triton)'],  # 图例名称
        styles=[('blue', '-'), ('red', '-'), ('green', '-'), ('orange', '-')],  # 线型和颜色
        ylabel='Latency (ms)',  # y轴标签
        plot_name='rope-performance',  # 图表标题
        args={
            'max_position': 8192,
            'base': 10000,
        },
    )
]


@triton.testing.perf_report(configs)
def benchmark(seed, batch_size, provider, seq_len=512, num_heads=8, head_size=128,
             rotary_dim=32, dtype=torch.float32, device='cuda',
             max_position=8192, base=10000, is_neox_style=True):
    """RoPE kernel benchmark function"""
    # 设置随机种子
    seed_everything(seed)

    # 准备输入张量
    positions = torch.randint(0, max_position, (batch_size, seq_len),
                            device=device)
    query = torch.randn(batch_size,
                       seq_len,
                       num_heads * head_size,
                       dtype=dtype,
                       device=device)
    key = torch.randn_like(query)
    
    scaling_factors = [1, 2, 4, 8]
    
    # 清空缓存
    _ROPE_DICT.clear()
    
    if provider in ['batched', 'batched-triton']:
        if provider == 'batched-triton':
            os.environ["VLLM_PREFER_TRITON_OPS"] = "1"
        # 准备 batched 版本
        batched_rope = get_rope(head_size, rotary_dim, max_position, base,
                               is_neox_style, {
                                   "type": "linear",
                                   "factor": tuple(scaling_factors)
                               })
        
        offset_map = torch.tensor(
            list(
                accumulate([0] + [
                    max_position * scaling_factor * 2
                    for scaling_factor in scaling_factors[:-1]
                ])), device=device)
        query_types = torch.randint(0,
                                  len(scaling_factors), (batch_size, seq_len),
                                  device=device)
        query_offsets = offset_map[query_types]
        flatten_offsets = query_offsets.flatten()

        # 定义测试函数
        def run_batched():
            batched_rope.forward(positions, query, key, flatten_offsets)

        # 运行基准测试
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run_batched, quantiles=quantiles)
        if provider == 'batched-triton':
            os.environ.pop("VLLM_PREFER_TRITON_OPS", None)

    else:
        if provider == 'non-batched-triton':
            os.environ["VLLM_PREFER_TRITON_OPS"] = "1"
        # 准备 non-batched 版本
        non_batched_ropes = [
            get_rope(head_size, rotary_dim, max_position, base, is_neox_style, {
                "type": "linear",
                "factor": (scaling_factor,)
            }) for scaling_factor in scaling_factors
        ]
        
        query_types = torch.randint(0,
                                  len(scaling_factors), (batch_size, seq_len),
                                  device=device)
        queries = [query[query_types == i] for i in range(len(scaling_factors))]
        keys = [key[query_types == i] for i in range(len(scaling_factors))]
        packed_qkr = list(zip(queries, keys, non_batched_ropes))

        # 定义测试函数
        def run_non_batched():
            for q, k, r in packed_qkr:
                r.forward(positions, q, k)

        # 运行基准测试
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run_non_batched, quantiles=quantiles)
        if provider == 'non-batched-triton':
            os.environ.pop("VLLM_PREFER_TRITON_OPS", None)

    return ms, max_ms, min_ms


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark the rotary embedding kernels.")
    parser.add_argument("--no-neox-style", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--rotary-dim", type=int, choices=[16, 32], default=32)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["bfloat16", "float"],
                        default="float")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device",
                        type=str,
                        choices=["cuda:0", "cuda:1"],
                        default="cuda:0")
    args = parser.parse_args()
    print(args)

    # 运行基准测试
    benchmark.run(
        show_plots=False,
        print_data=True,
        save_path="benchmark_rope_res",
        is_neox_style=not args.no_neox_style,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        rotary_dim=args.rotary_dim,
        dtype=getattr(torch, args.dtype),
        seed=args.seed,
        device=args.device,
    )
