### V4.1 独立训练基线与融合栈

V4.1 的模型、图像路由、压缩 attention、metadata 和并行化均由 `torchtitan_npu/models/deepseek_v4_1` 持有，不依赖 V4 模型或其专属 override。模型默认算子是 Attention Gym 的 eager `selected_attention`（`CompressedSparseInnerAttention2`）、公共 MoE 工厂与上游默认的 indexer 蒸馏损失 `IndexerDistillLoss`（coeff=0.01）。

当前支持 **FSDP + EP、TP1 / CP1 / PP1、eager 执行**，保留 FullAC 与图文输入。算子选择按具名配方（hardware recipes）：`*_multimodal` 为纯 reference，`*_multimodal_a3` 在 reference 上启用已验收融合（RMSNorm / 文本与视觉 RoPE / mHC post），`*_multimodal_a5` 在 A3 栈上追加 A5 专属的 sparse attention 与 mHC Sinkhorn。routed experts 的 grouped GEMM 是 reference 与融合共用的公共路径，不再作为独立融合开关。不支持量化、ngram、MTP、GraphTrainer；不支持的 TP、CP、PP 和 compile 配置在入口拒绝。

单机 8 卡入口（默认读取仓内上游 CC12M 测试 tar；数据源可通过标准 CLI 覆盖）：

```sh
# A3 融合栈（默认配方）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir>

# A5（A3 融合栈 + A5 专属算子）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh \
    --hf-assets-path <v41_tokenizer_dir>

# 纯 reference 对照（同一数据与并行形状）
CONFIG=deepseek_v4_1_flash_40layers_16experts_multimodal \
    bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir>

# 短跑（如 3 步）：显式同步步数与 LR 调度
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir> \
    --training.steps 3 --lr-scheduler.total-steps 3 --lr-scheduler.warmup-steps 2
```

A5 入口提供默认 CPU 亲和性布局，按主机拓扑通过 `CPU_AFFINITY_CONF` 覆盖。精度和性能验收记录见 PR 822（对应旧模型结构的历史证据；新结构下的对照按迁移后 reference 重新建立）。算子选择来自具名配方中的 `override.imports`，随 Trainer Config 打印；reference 对照使用 `*_multimodal`。

入口默认关闭 checkpoint；需要保存/加载时显式配置。tokenizer 统一经 `--hf-assets-path` 提供（测试可用仓内 `tests/assets/deepseek_v3` mini tokenizer）。

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

上游多模态模块需要与 PyTorch 匹配的 `torchvision`（CPU 图像预处理）；按运行环境安装对应版本。训练脚本只选择配方和运行环境，其他参数均透传标准 CLI。

### 相同初始权重（生成与加载）

baseline 与融合组要从同一份显式权重出发时，先生成再加载（dcp 格式，model-only）：

```sh
# 生成：跑 3 步保存训练 checkpoint（后续仅加载其中的模型权重）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir> \
    --training.steps 3 --lr-scheduler.total-steps 3 --lr-scheduler.warmup-steps 2 \
    --checkpoint.enable \
    --checkpoint.folder <ckpt_dir>

# 加载同一份权重起跑（--checkpoint.load-only 禁用后续保存）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir> \
    --checkpoint.enable \
    --checkpoint.initial-load-path <ckpt_dir>/step-3 \
    --checkpoint.initial-load-model-only \
    --checkpoint.load-only
```

两个要点：`--checkpoint.enable` 是加载生效的前提（不传时 CheckpointManager 未激活，会静默 fresh start，日志中只有一条 warning）；`initial-load-path` 指向 `step-N` 子目录。加载后应核对 checkpoint 日志与实际数据配置，再比较训练轨迹；单凭首步 loss 接近不能证明权重和数据一致。

### 融合算子逐项说明

融合栈由 `*_multimodal_a3`/`_a5` 配方默认启用（纯 reference 对照用 `*_multimodal` 配方）。逐项语义：

| override | 替换范围 | 实现 |
|---|---|---|
| `common.rms_norm.asc` | decoder、compressor、indexer 与视觉塔的公共 RMSNorm（保留各位置原有 epsilon） | Ascend RMSNorm |
| `common.rope.asc_workaround` | attention/compressor/indexer 的 split-aware 文本旋转（主干 builder 构造的 `WorkaroundComplexRoPE.Config` 的窄桥，保留 split/theta/YaRN） | Ascend rotary mul |
| `common.rope.asc_half_rotation` | 视觉塔 2D 位置表的 half 旋转（半宽 cos/sin，融合前复制为全宽表，逐 batch 折叠保位置） | 同上 |
| `sparse_attn.asc`（仅 A5） | `CompressedSparseInnerAttention2._compute_attention`：A5 TND kernel 返回完整 softmax 的输出与 LSE，蒸馏损失仍由模型注入 | Ascend sparse flash MLA |
| `mhc.asc_hc_post` | mHC post 变换 | Ascend mHC post |
| `mhc.asc_sinkhorn`（仅 A5） | `HcPre._split_sinkhorn`（新接口，`hc_eps` 字段） | npu_mhc_sinkhorn |

routed experts 已走公共 grouped GEMM（`_ClampGroupedExperts` → 仓内 NPU `_grouped_mm`），reference 与融合共用，不改变模型参数名称、FSDP/EP 或 checkpoint 格式。融合输出、输入/权重梯度与训练轨迹的误差需与迁移后同配置 reference 对照验收；仅运行成功不代表数值或性能验收完成。

### RoPE seam 说明

主线的文本旋转是 `WorkaroundComplexRoPE.Config`（带 split 前缀宽度）：reference 通过 `common.rope.workaround` 用展开实数表；融合配方通过 `common.rope.asc_workaround` 复用既有 NPU 实现（该桥为 Workaround 配置的精确匹配窄桥，`asc_complex` 仍只匹配上游 `ComplexRoPE.Config`，两者互不影响）。视觉塔的二维位置表使用 `HalfRotation` / `asc_half_rotation`，与文本 interleaved 布局不同，不能互换。inverse（attention o_rope 路径）经 `-sin` 支持。

CPU 测试覆盖两类 seam 的输出/梯度、partial 非连续输入、逐 batch 位置与配置隔离；CPU mock 不代表 NPU kernel 验证通过。

### Muon 优化器注意事项

`--optimizer.name Muon` 启用 DistMuon 时，`materialize()` 会整体替换 `param_groups` 为 `[DistMuon(pattern), AdamW(.*)]`，launcher 传入的 `--optimizer.param-groups.0.*` 参数静默失效。学习率等超参请使用顶层字段：`--optimizer.lr`、`--optimizer.weight-decay`、`--optimizer.muon-momentum` 等。indexer 参数在 LI 下有梯度，但仍留在 AdamW fallback 组，不因迁移改变参数策略。

### 集成与冒烟

V4.1 不保留专门 loss 锚（对齐上游 torchtitan 的模型测试方式），也不将 30 步用例留在默认 `models` 冒烟池；reference 路径的真实训练执行与融合冒烟均通过显式 suite 手动运行：

```bash
# 2 卡 CC12M + A3 融合 + Muon 两步冒烟（完成性检查，非精度验收）
python -m tests.integration_tests.run_tests /tmp/dsv41_fused --test_suite deepseek_v4_1_fused --ngpu 2
```

A5 使用 examples 下的 A5 训练脚本手动验证，不注册 A3 冒烟环境无法执行的 suite。
