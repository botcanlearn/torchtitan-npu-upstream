# 自定义 Context Parallel 特性

在分布式训练任务中，上下文并行（Context Parallelism, CP）是突破单卡内存瓶颈、
支持超长序列训练的核心技术。在基于 NPU 硬件生态推进 torchtitan 框架适配时，
现有技术方案暴露出显著的局限性：

1. PyTorch 原生 CP 设计强绑定于标准 SDPA 算子，仅提供 RingAttention 或原生
   AllGatherKV 的特定实现，无法覆盖 DeepSeek-V3.2 DSA
   （DeepSeek Sparse Attention）等稀疏注意力路径。
2. 框架需要允许开发者灵活扩展新的 CP 范式，如 Ulysses CP。

## 实现原理

`torchtitan_npu` 在
`torchtitan_npu/patches/distributed/custom_context_parallel.py` 中扩展
`torchtitan.distributed.context_parallel.apply_cp_to_attention_module`，
通过 `attention_type` 将 CP 路由到 NPU 自定义实现，同时保留上游 CP 的默认路径。
目前已提供如下两个自定义 CP context：

### SDPA Ulysses CP

定义在 `torchtitan_npu/distributed/context_parallel/ulysses_cp.py`，为常用的
SDPA 实现 Ulysses 风格的 CP。通过自定义 Context Parallel Context，我们在
基于 `torch_npu` 提供的融合算子的 Attention 计算前后插入 All-to-All 通信算子，
将数据从“序列并行分布”转换为“多头维度分布”，使单计算节点获得完整上下文。

### DSA CP

定义在 `torchtitan_npu/distributed/context_parallel/dsa_cp.py`，为
DeepSeek Sparse Attention 提供 AllGatherKV 风格的 CP。本项目在遵循
torch 原生 CP 设计逻辑的基础上，通过自定义 Context Parallel Context，
将注意力部分的 forward 函数替换为 CP 感知实现，对 KV 相关激活做 CP 域的
AllGather，并在反向传播中完成对应的 ReduceScatter，以确保 DSA 注意力在
CP 场景下的正确性。

关于 DSA CP 的更多原理介绍，参考
[技术文档](https://gitcode.com/cann/cann-recipes-train/blob/master/docs/llm_pretrain/deepseekv32_pre_train_optimization.md#自定义CP策略)。

## 支持范围

| 模型 | CP 实现 | 触发条件 | 说明 |
| --- | --- | --- | --- |
| DeepSeek-V3 | Ulysses CP | `context_parallel_degree > 1` 且 `enable_custom_context_parallel=True` | SDPA 路径使用 All-to-All 变换 |
| DeepSeek-V3.2 | DSA CP | `context_parallel_degree > 1` 且启用 `npu_dsa` converter | - |


## 配置选项

Custom CP 配置写在模型的 `config_registry.py` 中。对应字段
`torchtitan_npu.config.configs.ParallelismConfig`。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `context_parallel_degree` | int | 1 | Context 并行度；大于 1 时启用 CP mesh |
| `enable_custom_context_parallel` | bool | False | 启用 NPU 自定义 CP 路径；DeepSeek-V3.2 长序列配置中已显式开启 |

> **注意**：DeepSeek-V3.2 的 DSA CP 依赖 `npu_dsa` converter。
> 若 `context_parallel_degree > 1` 但 converters 中没有 `npu_dsa`，
> 并行化阶段会报错，提示 `CP on deepseek_v32 requires 'npu_dsa' converter`。

CP类型由框架根据模型及转换器配置**自动选择**，无需手动指定：

- **DeepSeek-V3**：启用自定义 CP 后自动使用 Ulysses CP。
- **DeepSeek-V3.2**：当 converters 中包含 `npu_dsa` 且 CP 开启时，自动使用 DSA CP。

### 配置示例一：SDPA Ulysses CP（DeepSeek-V3）

在 `torchtitan_npu/models/deepseek_v3/config_registry.py` 的配置函数中设置：

```python
from torchtitan_npu.config.configs import ParallelismConfig

parallelism = ParallelismConfig(
    context_parallel_degree=2,
    enable_custom_context_parallel=True,
)
```

启动时也可以直接覆盖：

```bash
bash scripts/run_train.sh \
  --parallelism.context_parallel_degree 2 \
  --parallelism.enable_custom_context_parallel
```

### 配置示例二：DSA Context Parallel（DeepSeek-V3.2）

DeepSeek-V3.2 `config_registry.py` 中的
`deepseek_v32_671b_61layers_32k_128die` 配置默认启用 32k 长序列 CP：

```python
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import ParallelismConfig
from torchtitan_npu.converters.npu_registry import get_model_converter_config

model_converters = ModelConvertersContainer.Config(
    converters=[
        get_model_converter_config("npu_dsa"),
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_permute"),
        get_model_converter_config("npu_gmm"),
    ],
)
parallelism = ParallelismConfig(
    context_parallel_degree=8,
    enable_custom_context_parallel=True,
)
```

多机长序列训练推荐在 `scripts/run_train_multinodes.sh` 中配置节点 IP、
`NPUS_PER_NODE` 等集群信息后，直接覆盖为 32k 预置配置启动：

```bash
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_61layers_32k_128die \
bash scripts/run_train_multinodes.sh
```

调试时也可以基于已有配置覆盖 CP 参数，例如：

```bash
NGPU=16 \
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_4layers_debug \
bash scripts/run_train.sh \
  --parallelism.context_parallel_degree 2 \
  --parallelism.enable_custom_context_parallel
```

上述覆盖方式仍要求 DeepSeek-V3.2 配置中保留 `npu_dsa` converter。

## 约束与注意事项

- DeepSeek-V3.2 CP 当前仅支持 `attn_type="sdpa"` 的入口，并在 CP 路径中根据
  `npu_dsa` converter 自动切换到 DSA CP 路由。
- `training.seq_len` 需要能被 TP 与 CP 共同切分。当前实现中会校验
  `training.seq_len % parallel_dims.seq_len_divisor == 0`，报错信息会提示
  `seq_len` 需要能被 `TP degree * 2 * CP degree` 整除。
- DeepSeek-V3.2 DSA CP 会对 `block.attention.inner_attention` 应用 CP patch，
  并将 `model_args`、`tp_mesh` 和 converters 传入自定义 CP 路由。
- 若使用 32k 长序列训练，建议优先从
  `deepseek_v32_671b_61layers_32k_128die` 预置配置开始调整，避免遗漏 DSA
  converter、CP 并行度和长序列 batch 配置。
