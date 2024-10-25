import os
import time
from typing import Optional, Tuple

import torch
import triton
import triton.testing

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser, seed_everything

def create_benchmark_configs(dtypes: list = None):
    if dtypes is None:
        dtypes = [torch.half, torch.bfloat16, torch.float]
    
    configs = []
    for dtype in dtypes:
        for add_residual in [True, False]:
            dtype_str = str(dtype).split('.')[-1]
            if add_residual:
                plot_name = f'rms-norm-performance-residule-{dtype_str}'
            else:
                plot_name = f'rms-norm-performance-{dtype_str}'
            configs.append(
                triton.testing.Benchmark(
                    x_names=['num_tokens'],  # 横轴参数
                    x_vals=[512, 1024, 2048, 4096, 8192],  # 测试更多的序列长度
                    line_arg='provider',  # 用于区分不同线条的参数
                    line_vals=['cuda', 'triton'],  # 不同provider的取值
                    line_names=['CUDA', 'Triton'],  # 图例中显示的名称
                    styles=[('blue', '-'), ('green', '-')],  # 线条样式
                    ylabel='Latency (ms)',  # y轴标签
                    plot_name=plot_name,  # 为不同数据类型创建不同图表
                    args={
                        "dtype": dtype,
                        "add_residual": add_residual
                    }
                )
            )
    return configs


@triton.testing.perf_report(create_benchmark_configs())
def benchmark_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    provider: str,
    seed: int = 0
) -> Tuple[float, float, float]:
    seed_everything(seed)
    torch.set_default_device("cuda")

    # 设置是否使用Triton实现
    os.environ["VLLM_PREFER_TRITON_OPS"] = "1" if provider == 'triton' else "0"

    # 初始化模型和输入数据
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # 使用lambda包装前向传播
    forward_fn = lambda: layer(x, residual)

    # 使用triton的benchmark工具进行测试，获取多个分位数的结果
    quantiles = [0.5, 0.2, 0.8]  # 中位数、20%分位数和80%分位数
    ms, min_ms, max_ms = triton.testing.do_bench(forward_fn, quantiles=quantiles)

    return ms, min_ms, max_ms

def main(
    hidden_size: int,
    seed: int = 0
) -> None:
    # 运行benchmark并生成性能报告
    benchmark_rms_norm.run(
        print_data=True,  # 打印详细数据
        save_path='benchmark_layernorm_res',
        hidden_size=hidden_size,
        seed=seed
    )

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark the layernorm kernel.")
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    print(args)

    main(
        hidden_size=args.hidden_size,
        seed=args.seed
    )