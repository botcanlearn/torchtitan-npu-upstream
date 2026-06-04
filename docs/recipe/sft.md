# SFT 指令微调

torchtitan-npu 支持基于对话数据的指令微调（Supervised Fine-Tuning, SFT）。当前已提供 Qwen3-30B-A3B 模型在 NPU 上的开箱即用 SFT 配置，支持 Ulysses Context Parallel 长序列训练。

## 快速开始：Qwen3-30B-A3B SFT

### 前置条件

- 准备 Qwen3-30B-A3B 的 HuggingFace 预训练权重（如 `./assets/hf/Qwen3-30B-A3B`）
- 准备训练数据集（详见[数据预处理](#数据预处理)）

### 一键启动

```bash
MODULE=torchtitan_npu.models.qwen3 CONFIG=sft_qwen3_30ba3b_math bash scripts/run_train.sh
```

## 数据预处理

SFT 的数据预处理在 `torchtitan_npu/models/qwen3/config_registry.py` 的配置函数中完成。以 `sft_qwen3_30ba3b_math` 为例，它使用 GSM8K 数据集，`process_sample` 将原始样本转为 `[user, assistant]` 消息列表：

```python
def sft_qwen3_30ba3b_math() -> Trainer.Config:

    def process_sample(sample):
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        return [
            {"role": "user", "content": sample["question"]},
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": final_answer.strip(),
            },
        ]

    # ... 其余配置 ...
    dataloader=ChatDataLoader.Config(
        dataset_path="openai/gsm8k",
        load_dataset_kwargs={"name": "main", "split": "train"},
        sample_processor=process_sample,
    )
```

框架拿到 `process_sample` 返回的消息列表后，自动完成以下处理：
1. 调用 tokenizer 的 chat template 将消息渲染为完整文本
2. 通过增量前缀重 tokenize 定位 prompt/response 边界，对 prompt 部分做 label mask（仅对 assistant 回复计算 loss）
3. 对多个样本做 greedy packing（将短样本打包到同一 seq_len 窗口，EOS 分隔，per-document position 重置）
4. 超过 seq_len 的样本自动丢弃（而非截断）

### 消息格式

`process_sample` 需返回如下格式的消息列表：

```python
[
    {"role": "user", "content": "用户输入"},
    {"role": "assistant", "content": "助手回复"},
]
```

Qwen3 支持 thinking mode，可在 assistant 消息中添加 `reasoning_content` 字段：

```python
[
    {"role": "user", "content": "用户输入"},
    {"role": "assistant", "reasoning_content": "思考过程", "content": "最终回答"},
]
```

> **限制**：当前仅支持单轮对话（一条 user + 一条 assistant），暂不支持多轮。

## Ulysses Context Parallel

torchtitan-npu 为 Qwen3 提供了 Ulysses 风格的 CP 实现。在配置中设置 `context_parallel_degree > 1` 即可启用：

```bash
bash scripts/run_train.sh --parallelism.context_parallel_degree 4
```

关于 Ulysses CP 的实现原理和约束条件，详见[自定义 Context Parallel 特性文档](../feature_guides/parallelism/custom_cp.md)。

## 权重加载与保存

### 加载预训练权重

SFT 配置默认从 HuggingFace 格式加载预训练权重：

```bash
--checkpoint.initial_load_in_hf \
--checkpoint.initial_load_path /path/to/Qwen3-30B-A3B
```

> **注意**：`checkpoint.folder` 不能与 `checkpoint.initial_load_path` 相同，否则框架会跳过 HuggingFace 权重加载。

### 切换为自己的预训练权重

修改 `checkpoint.initial_load_path` 和 `hf_assets_path` 指向新的权重目录：

```bash
--checkpoint.initial_load_path /path/to/your/model \
--hf_assets_path /path/to/your/model
```

### 保存训练后的权重

训练完成后，权重会自动保存到 `checkpoint.folder` 指定的路径。可通过 `checkpoint.interval` 设置保存间隔：

```bash
--checkpoint.folder /path/to/save \
--checkpoint.interval 100
```

如仅需加载权重而不保存（如调试时），设置 `--checkpoint.load_only`。

## 常见问题

**Q: 数据集跑完一轮就停了？**

默认 `dataloader.infinite=true`，数据集会无限循环。如果设为 false，数据遍历完一轮后训练会停止。tyro 的 bool 参数通过 flag 设置，显式开启或关闭可使用：

```bash
--dataloader.infinite
--dataloader.no-infinite
```

**Q: 超长样本被丢弃了？**

`ChatDataset` 会自动丢弃 token 数超过 `seq_len` 的样本（而非截断），日志中会打印 `Dropping sample` 提示。可增大 `seq_len` 或开启 Ulysses CP 来容纳更长样本。
