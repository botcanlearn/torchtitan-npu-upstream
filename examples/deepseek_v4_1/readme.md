### V4.1 独立训练基线与融合栈

V4.1 的模型、图像路由、压缩 attention、metadata 和并行化均由 `torchtitan_npu/models/deepseek_v4_1` 持有，不依赖 V4 模型或其专属 override。模型默认算子是 Attention Gym 的 eager `selected_attention`（`CompressedSparseInnerAttention2`）、公共 MoE 工厂与上游默认的 indexer 蒸馏损失 `IndexerDistillLoss`（coeff=0.01）。

当前支持 **FSDP + EP、TP1 / CP1 / PP1、eager 执行**，保留 FullAC 与图文输入。启动方式与 DSV4 一致：A3 脚本组织公共实验参数与融合列表，A5 调用 A3 并追加 CPU 亲和性、partial 文本 RoPE、两项 SwiGLUGroup、sparse attention 与 mHC Sinkhorn；两种入口均通过 `USE_GOLDEN=1` 选择 reference。A3 默认融合包含 RMSNorm、文本（`asc_complex`）与视觉 RoPE、MoE token dispatcher 和 mHC post；A5 将文本 RoPE 换为 `asc_partial` 并追加 routed/shared SwiGLUGroup、sparse 与 Sinkhorn。routed experts 的 grouped GEMM 是 reference 与融合共用的公共路径，不再作为独立融合开关。不支持量化、ngram、MTP、GraphTrainer；不支持的 TP、CP、PP 和 compile 配置在入口拒绝。

单机 8 卡入口（默认读取仓内上游 CC12M 测试 tar；数据源可通过标准 CLI 覆盖）：

```sh
# A3 融合栈（默认）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir>

# A5（A3 融合栈 + A5 专属算子）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh \
    --hf-assets-path <v41_tokenizer_dir>

# 纯 reference 对照（同一数据与并行形状）
USE_GOLDEN=1 \
    bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir>

# 短跑（如 3 步）：显式同步步数与 LR 调度
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir> \
    --training.steps 3 --lr-scheduler.total-steps 3 --lr-scheduler.warmup-steps 2
```

默认实验为 40 层 / 16 专家、seq512、local/global batch 1/8、FSDP8/EP8、AdamW、eager、FullAC，训练与调度均为 40 步。Python 配方保留模型结构、数据协议和优化器参数布局；实验参数由脚本组织，末尾 CLI 参数覆盖脚本默认值。`CONFIG` 可选择 `deepseek_v4_1_debugmodel_multimodal` 调试宽度，硬件选择仍由脚本负责。

A5 的 `CPU_AFFINITY_CONF` 应按主机拓扑覆盖。`CLI_OVERRIDES` 沿用 DSV4 的列表扩展方式：A3 基础列表之外，文本 RoPE 经该通道按硬件选择（A3 默认 `asc_complex`，A5 换为 `asc_partial`），A5 默认追加 SwiGLUGroup、sparse attention 与 Sinkhorn，不重复传入 `--override.imports`。显式传 `--override.imports` 会替换整个集合，需自行包含所需的 RoPE 与 swap optimizer。最终选择随 Trainer Config 打印。

普通运行不强制随机种子和确定性；精度对照须在双方命令中追加 `--debug.seed 42 --debug.deterministic`。默认关闭 checkpoint 且设置 `load_only=True`；保存时同时传 `--checkpoint.enable --checkpoint.no-load-only`。`load_only` 表示禁止保存，与 model-only 加载不同。tokenizer 统一经 `--hf-assets-path` 提供（测试可用仓内 `tests/assets/deepseek_v3` mini tokenizer），该参数本身不加载模型权重。

indexer 蒸馏损失由 `IndexerDistillLoss` 实现（上游默认 `coeff=0.01`）：每层只要消费了 selection 就挂一个损失，教师用该层自身 attention 的完整 softmax 分母（窗口 + 压缩条目 + sink）重建，按压缩切片的边际质量加权；梯度经 `_AuxLossInjection` 注入，只训练 indexer 自身参数，训练指标为 `indexer_distill_loss/mean`。打包 loader 标记的结构 padding 行不参与蒸馏（真实图像 token 保留训练资格）。

### 真实 CC12M 数据入口（图像条件 caption 预测）

数据入口与 GitHub TorchTitan 的 Qwen/Kimi 多模态训练一致，使用 CC12M WebDataset tar（同名 `.jpg` / `.txt` 样本对）。默认 `dataset=cc12m-test`，读取仓内 `tests/assets/cc12m_test/`，无需传入 dataloader 参数；其他本地 tar 目录通过 `--dataloader.dataset-path <cc12m_tar_dir>` 指定。不提供 manifest、离线准备或 digest 专用脚本。

数据准备沿用上游的两种方式：

- 在线：选择 `--dataloader.dataset cc12m`，不指定 `dataset-path`，从 Hugging Face 流式读取，无需预先下载完整数据集。
- 本地：下载包含图片和 caption 的 WebDataset tar 分片，使用 `--dataloader.dataset-path <cc12m_tar_dir>`。不解压、不生成 manifest，不预先 tokenize 或计算 ViT 特征；图片预处理在数据加载时执行，ViT 在模型前向中执行。

`cc12m-train-0000.tar` 原样复用 [GitHub TorchTitan v0.3.0 测试资产](https://github.com/pytorch/torchtitan/blob/v0.3.0/tests/assets/cc12m_test/cc12m-train-0000.tar)，包含 32 条图文样本，仅用于测试。完整在线数据源通过 `--dataloader.dataset cc12m` 选择；指定本地目录后，读取的是该目录的数据，而非仓内测试资产。

直接复用上游 `HuggingFaceMultiModalDataset` 的 DP 分片、`MMSamplePacker` 和状态恢复，以及 `ParallelAwareDataloader`。针对 TorchTitan 0.3.0 的 HF 恢复起点跨 epoch 重放问题，按上游文本 loader 的方式补齐 `set_epoch`；packing 缓冲满时调用上游 `flush()`。DSV4.1 只适配协议层：`BOS + 完整图片协议 + caption + EOS`、每篇文档按模型压缩比的最小公倍数对齐（当前为 2，至多在 EOS 后补 1 个不监督的 pad）、`valid_tokens` 标记随 pack 消费并穿过 packer；只有 caption 与 EOS 参与监督，图片特征索引按 pack 中的图片顺序连续编号，每篇文档的位置从零开始，注意力和压缩块隔离文档边界。

与上游一致，`--dataloader.packing-buffer-size 0` 默认关闭 packing；设置正值（例如 `128`）启用。一个 rank 的 local batch 仍为 1，但一个 packed 序列可以包含多篇图文样本。装箱和合并仍调用上游实现；缓冲量还包括待输出的 packed 队列及 loader 预取，不是进程内存的硬上限。

A/B 必须固定 tar 内容及顺序、tokenizer、序列长度、packing buffer、DP/全局 batch、seed 和初始权重。旧模型结构的 manifest/合成轨迹与新入口、新 LI 训练目标的轨迹互不可比；迁移后的对照统一使用新 reference 起点或同一份完整 checkpoint。精确续训使用完整 checkpoint 恢复 loader 状态；model-only 加载只复用权重（新旧结构需按参数名对账后映射，不做静默兼容）。

上游多模态模块需要与 PyTorch 匹配的 `torchvision`（CPU 图像预处理）；按运行环境安装对应版本。数据路径支持 `DATASET_PATH` 环境变量或标准 CLI；不指定路径时使用所选数据集的上游注册源。

### 相同初始权重（生成与加载）

baseline 与融合组要从同一份显式权重出发时，先生成再加载（dcp 格式，model-only）：

```sh
# 生成：跑 3 步保存训练 checkpoint（后续仅加载其中的模型权重）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir> \
    --training.steps 3 --lr-scheduler.total-steps 3 --lr-scheduler.warmup-steps 2 \
    --checkpoint.enable --checkpoint.no-load-only \
    --checkpoint.folder <ckpt_dir>

# 加载同一份权重起跑（--checkpoint.load-only 禁用后续保存）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir> \
    --checkpoint.enable \
    --checkpoint.initial-load-path <ckpt_dir>/step-3 \
    --checkpoint.initial-load-model-only
```

两个要点：`--checkpoint.enable` 是加载生效的前提（不传时 CheckpointManager 未激活，会静默 fresh start，日志中只有一条 warning）；`initial-load-path` 指向 `step-N` 子目录。输出 `checkpoint.folder` 中已有有效 checkpoint 时会优先恢复它；从 `initial-load-path` 起跑应使用新的输出目录。加载后应核对 checkpoint 日志与实际数据配置，再比较训练轨迹；单凭首步 loss 接近不能证明权重和数据一致。

### 融合算子逐项说明

融合栈由 A3/A5 脚本默认启用，reference 使用同一入口的 `USE_GOLDEN=1`。逐项语义：

| override | 替换范围 | 实现 |
|---|---|---|
| `common.rms_norm.asc` | decoder、compressor、indexer 与视觉塔的公共 RMSNorm（保留各位置原有 epsilon） | Ascend RMSNorm |
| `common.rope.asc_complex`（A3 文本） | attention/compressor/indexer 的 split-aware 文本旋转（公共 `ComplexRoPE.Config`，保留 split/theta/YaRN） | Ascend rotary mul（interleave） |
| `common.rope.asc_partial`（A5 文本） | 同上三处文本位点；单个 `inplace_partial_rotary_mul` 只旋转尾部 `dim` 通道 | AscendC partial rotary |
| `common.rope.asc_half_rotation` | 视觉塔 2D 位置表的 half 旋转（半宽 cos/sin，融合前复制为全宽表，逐 batch 折叠保位置） | 同上 |
| `sparse_attn.asc`（仅 A5） | `CompressedSparseInnerAttention2._compute_attention`：A5 TND kernel 返回完整 softmax 的输出与 LSE，蒸馏损失仍由模型注入 | Ascend sparse flash MLA |
| `common.swiglu_group.asc`（仅 A5） | `*.moe.routed_experts.inner_experts` 的 grouped SwiGLU（FQN 限定） | cann_ops_nn.swiglu_group |
| `common.swiglu_group.asc_shared_experts`（仅 A5） | `*.moe.shared_experts` 的 SwiGLU（FQN 限定，不替换视觉 MLP） | 同上 |
| `common.token_dispatcher.asc` | 公共 MoE 的 permute / re-routing / unpermute；reference 不启用 | Ascend token dispatcher |
| `mhc.asc_hc_post` | mHC post 变换 | Ascend mHC post |
| `mhc.asc_sinkhorn`（仅 A5） | `HcPre._split_sinkhorn`（新接口，`hc_eps` 字段） | npu_mhc_sinkhorn |

routed experts 已走公共 grouped GEMM（`_ClampGroupedExperts` → 仓内 NPU `_grouped_mm`），reference 与融合共用，不改变模型参数名称、FSDP/EP 或 checkpoint 格式。融合输出、输入/权重梯度与训练轨迹的误差需与迁移后同配置 reference 对照验收；仅运行成功不代表数值或性能验收完成。

### RoPE seam 说明

文本旋转由主干构造公共 `ComplexRoPE.Config`（带 split 前缀宽度）：reference 通过 `common.rope.workaround` 用展开实数表；A3 融合选 `asc_complex`，A5 融合选 `asc_partial`，三者精确匹配同一配置类，同一列表只能出现一个（override 框架对同节点双 claim 直接报错）。视觉塔的二维位置表使用 `HalfRotation` / `asc_half_rotation`，与文本 interleaved 布局不同，不能互换。inverse（attention o_rope 路径）经 `-sin` 支持。

CPU 测试覆盖两类 seam 的输出/梯度、partial 非连续输入、逐 batch 位置与配置隔离；CPU mock 不代表 NPU kernel 验证通过。

### Muon 优化器注意事项

`--optimizer.name Muon` 启用 DistMuon 时，`materialize()` 会整体替换 `param_groups` 为 `[DistMuon(pattern), AdamW(.*)]`，launcher 传入的 `--optimizer.param-groups.0.*` 参数静默失效。学习率等超参请使用顶层字段：`--optimizer.lr`、`--optimizer.weight-decay`、`--optimizer.muon-momentum` 等。indexer 参数在 LI 下有梯度，但仍留在 AdamW fallback 组，不因迁移改变参数策略。

### 验证范围

CPU 单测覆盖模型与融合配置选择、RoPE 输出/梯度等核心行为；不保留专用的 V4.1 两卡手动冒烟 suite。真实训练通过上述 A3/A5 入口执行。历史 A3 的 50 步对照见 PR 845；该证据不代表当前 HEAD 或 A5 新 CANN 已完成重验。
