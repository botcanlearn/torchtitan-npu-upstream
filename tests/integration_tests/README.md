# 集成测试基础设施

本目录遵循 Torchtitan 的 `tests/integration_tests` 布局，负责维护集成测试定义、测试入口以及可选的 loss 精确比较。基础架构代码由
torchtitan 迁移而来。

默认 `models` suite 覆盖 DeepSeek-V4、DeepSeek-V3.2 和 Qwen3.5；checkpoint、量化及 Engram HF 另有独立 suite。

## 入口

CI 通过以下脚本启动本目录的 integration ST：

```bash
.ci/integration_test.sh
```

`.ci/smoke_test.sh` 只负责 `tests/smoke_tests` 的 smoke 阶段，不再触发本目录的 ST。
两个入口都先 source `.ci/common.sh`，由它完成 CANN 环境、解释器 shim 和 torchtitan
checkout 准备。

或直接运行 Python 入口：


```bash
python -m tests.integration_tests.run_tests \
  ./test_reports/integration \
  --test_suite models \
  --ngpu 4
```
其中`./test_reports/integration` 是必填的测试输出目录，运行前需要确保该目录为空。

直接运行上述 Python 命令仅执行 integration tests。`--test_suite models` 与 CI 的集成测试配置保持一致，覆盖 DeepSeek-V4 和 DeepSeek-V3.2。完整 CI 流程还会在此之前执行 `tests/smoke_tests`。

## 主要测试矩阵

| Case 名称 | 模型 | 并行配置 | Rank 数 | 编译配置 | Check Loss | 不检查 Loss 原因 |
|---|---|---|---|---:|---|---|
| `dsv4_anticipatory_recovery` | DeepSeek-V4 | 1 Rank、原生 checkpoint 回滚 | 1 | - | 否 | 测试入口一次性扰动真实训练 loss，断言 override 和完整恢复周期 |
| `dsv4_golden_1rank` | DeepSeek-V4 | 1 Rank 参考配置 | 1 | - | 是 | - |
| `dsv4_golden_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | - | 是 | - |
| `dsv4_muon_swap_ep2_fsdp2` | DeepSeek-V4 | NPU 融合算子 + DistMuon/AdamW NovaSwap、EP2 + FSDP2、2 steps | 2 | - | 否 | 两步训练 smoke；未生成 swap 数值 golden，也未单独断言 swap action |
| `dsv4_lora_ep2_fsdp2` | DeepSeek-V4 LoRA | EP2 + FSDP2，固定基座，训练 LoRA A/B，2 steps | 2 | - | 否 | 检查冻结参数与 routing bias 不变、adapter 更新及最终 PEFT 导出；不做中间保存或恢复 |
| `dsv4_checkpoint_resume_ep2_fsdp2` | DeepSeek-V4 | Golden recipe + AdamW NovaSwap、EP2 + FSDP2，step 2 恢复到 step 4 | 2 | - | 否，动态精确比较 loss/grad_norm | 与本次连续训练的 step 3、4 精确比较；不读取仓内 golden loss |
| `dsv4_smla_1rank_aot_eager` | DeepSeek-V4 | 1 Rank | 1 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_smla_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_smla_cp2_ep2_fsdp2` | DeepSeek-V4 | CP2 + EP2 + FSDP2 | 4 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_mtp_smla_cp2_headtail` | DeepSeek-V4 MTP | CP2 + headtail | 2 | - | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv3_2_dsa_1rank` | DeepSeek-V3.2 | 1 Rank，DSA | 1 | - | 是 | - |
| `dsv3_2_dsa_ep2_fsdp2` | DeepSeek-V3.2 | DSA + EP2/FSDP2 | 2 | - | 是 | - |
| `dsv3_2_dsa_cp2` | DeepSeek-V3.2 | DSA + CP2 | 2 | - | 否 | ST 仅验证训练触发；CPU metadata oracle 单独覆盖，暂未生成 CP2 golden loss |
| `dsv4_ema_ep2_fsdp2` | DeepSeek-V4 | Golden + EP2/FSDP2 + EMA CPU offload | 2 | - | 否 | 校验完整 DCP metadata 包含 `ema_optimizer.*` |

`use_golden` 与 `check_loss` 是两个独立维度：`use_golden` 仅决定使用 Golden 参考算子
还是 SMLA/NPU override；`check_loss` 决定是否启用 deterministic、读取参考 loss 并执行
精确数值比较。

当前两个 Golden case（均为 V4）设置 `check_loss=True`，使用固定随机种子和 deterministic 模式，
比较 TensorBoard 标量 `loss_metrics/global_avg_loss`，要求 step 集合和每个浮点值均精确相等。
DeepSeek-V4.1 的常规训练验证使用 [A3/A5 示例入口](../../examples/deepseek_v4_1/readme.md)。Engram HF 验证使用下述独立 suite，不在默认 `models` 或 CI smoke 中执行。

两个 DeepSeek-V3.2 case 同样设置 `check_loss=True`，使用 RoPE workaround、Ascend DSA
metadata/attention override，并分别对 1-rank 和 EP2/FSDP2 的 100-step loss 做精确比较。

`dsv4_checkpoint_resume_ep2_fsdp2` 覆盖 AdamW NovaSwap 的 checkpoint 保存、恢复和精度对齐，已注册到
独立的 `deepseek_v4_checkpoint` suite，不在默认 `models` suite 中执行。它使用两卡 EP2 + FSDP2、Golden recipe、`swap_optimizer` override 和
`--optimizer.name=AdamW`，设置 `use_golden=True` 与 `check_resume=True`，
固定 seed=42 并开启 deterministic。第一阶段连续训练 4 步，保留 step 2 的完整 checkpoint；
第二阶段在新进程中通过 `--checkpoint.load-step=2` 恢复，再训练第 3、4 步。
两阶段均设置 `--training.steps=4`，确保学习率调度一致，共用 checkpoint 目录，
分别写入 `tb_phase_0` 和 `tb_phase_1`。检查 TensorBoard 的
`loss_metrics/global_avg_loss` 和 `grad_norm`：步骤集合必须分别为 `(1, 2, 3, 4)` 和 `(3, 4)`，
续训两步的两个标量必须与连续训练逐值精确相等，不舍入、不使用容差；缺失、重复步骤或
非有限值均失败。本次连续训练是动态基准，不读取或更新仓内 golden loss 文件。

单独执行此用例：

```bash
python -m tests.integration_tests.run_tests /tmp/checkpoint_resume_output \
  --test_suite deepseek_v4_checkpoint --test_name dsv4_checkpoint_resume_ep2_fsdp2 --ngpu 2
```

`dsv4_lora_ep2_fsdp2` 在实际模型并行化后冻结基座、训练 LoRA A/B。
连续训练 2 步，检查冻结参数与 routing bias 不变、可训练 adapter 更新。
最后一步通过实际 checkpointer 导出 PEFT，检查 adapter 文件的 key、shape、数值和配置中的 rank、alpha、target。
该用例不做中间 checkpoint 保存或恢复。冻结 LoRA A、仅训练 B 的差异由 CPU 单元测试覆盖。

四个 SMLA case 都设置 `check_loss=False`，因此不会启用 `--debug.deterministic`，也不会
读取 golden loss。它们用于覆盖 SMLA/NPU override 在单卡、EP+FSDP、CP+EP+FSDP 以及
MTP+CP 场景下的实际构图、编译和训练执行路径；单卡、EP2 和 CP2+EP2 场景均使用
`aot_eager`，并默认覆盖 fused MoE token dispatcher。MTP+CP 用例固定使用
`deepseek_v4_debugmodel`、CP2 和 headtail，在 C4 packed sequence 上执行完整的
MTP forward、chunked loss 和 backward。

`dsv4_muon_swap_ep2_fsdp2` 保留 NPU 融合算子 recipe：Ascend RMSNorm、complex RoPE、sparse
attention、MHC 和 MoE token dispatcher；两卡 EP2/FSDP2，显式选择 `--optimizer.name=Muon`
并启用 `swap_optimizer` override。它运行两步，覆盖 DistMuon momentum state 与其 AdamW fallback
state 的 NovaSwap 路径。该 case 同样只检查训练完成，不读取 golden loss，也不声称数值等价。

这里的 integration recipe 聚焦 sparse-attention / MHC 回归边界。端到端 example 脚本
额外启用 Virtual Optimizer；checkpoint 保存兼容由 extension `CheckpointManager` 提供。
这些 optimizer state/checkpoint 路径不属于当前 integration loss regression 的覆盖范围。

## Nightly All Models（A3 / A5 CI 用例）

这里维护**测试定义与训练验收语义**；GitHub Actions 的触发、授权、部署、资源锁、多用例调度、结果查询和故障排查统一参阅 [Lite Actions 使用与开发指南](https://github.com/depeng1994/lite-actions/blob/main/README.md)，不在两仓重复维护。

- 用例：[`nightly_all_models_test/`](nightly_all_models_test/) 中的 `a3_8p_tests.py`、`a3_16p_tests.py`、`a5_64p_tests.py`；各模块通过 `build_test_list()` 返回 `OverrideDefinitions`。
- 通用执行：[`nightly_all_models_test/runner.py`](nightly_all_models_test/runner.py)；Lite Actions 专属适配入口：[`tools/lite_actions/entrypoint.py`](tools/lite_actions/entrypoint.py)。测试定义不放进 Lite Actions 目录，也不维护第二份 `ci_registry.json`。
- `override_args` 声明与共享 example 不同的训练 CLI；不要复制 example 的 NPU imports。当前 `env_vars` 承载用例选定的 CANN/HF/Checkpoint 资产及必要模型环境；执行机的 SSH 地址、物理 NPU 分配和 HCCL 拓扑由 Lite Actions 管理。

| Suite / Case | 训练语义 | 状态 |
| --- | --- | --- |
| `a3_8p_tests` / `dsv4_flash_a3_8p_example` | A3 单机 8P、Muon、Eager、5 steps | 需以本次代码的新 Run 验收 |
| `a3_8p_tests` / `dsv4_flash_a3_8p_adamw` | 同模型与卡数，AdamW（禁用默认 optimizer swap）、Eager、5 steps | 需以本次代码的新 Run 验收 |
| `a3_16p_tests` / `dsv4_flash_a3_16p_example` | A3 双机 16P、AdamW、EP16、Eager、5 steps | 有历史 Eager 成功；本次修改仍需回归 |
| `a5_64p_tests` / `dsv4_pro_a5_64p` | A5 八机 64P、DeepSeek-V4 Pro | 通道禁用，资产/网络及实机训练未验收 |

上述 Eager 测试**不代表** Inductor、数值 golden 或 A5 训练已通过。两个 8P case 验证的是不同优化器路径，不是用不同 steps 制造重复测试。训练步数固定在 testcase 的 `--training.steps` CLI 中，当前使用 `expected_steps` 核对 TensorBoard 记录；GitHub 输入仅选择 `test_id` 或 `suite`，不接受可变 `STEPS`。

新增或修改 case：直接编辑对应 `*_tests.py` 的 `build_test_list()`，保持唯一 `test_name`、正确的 `ngpu/nnodes`、训练入口与 CLI，并按需设置 `env_vars`、验收字段。复用现有资源通道时不需要修改 Lite Actions 注册表。**选择用例、触发、预检查和验收步骤**详见上述 Lite Actions README。

## 单机 Integration Runner 的并行执行

`python -m tests.integration_tests.run_tests` 复用本仓基于上游 TorchTitan GPUPool 的 NPU 并发机制：使用实际可见 NPU 建池，以 `ASCEND_RT_VISIBLE_DEVICES` 隔离用例；资源不足的用例明确 skip。需要逐个执行时传 `--no-parallel`。每个 case 的完整输出按名称归档，训练进程失败或超时会导致测试失败并清理所属子进程组。

此处描述的是**单机 Integration Runner**，并非 Lite Actions 的跨机器 SSH 调度。多机资源锁、外部进程占用保护及完整日志位置请以 [Lite Actions README](https://github.com/depeng1994/lite-actions/blob/main/README.md) 为准。同机并发运行独立 HCCL 任务时还需避免通信端口冲突。

## LoRA training and resume

The default `models` suite includes `dsv4_lora_ep2_fsdp2` for training and PEFT export. The distributed DCP resume case, `dsv4_lora_resume_ep2_fsdp2`, runs separately in `deepseek_v4_checkpoint` (two NPUs each). The resume case compares optimizer state exactly before the first resumed update and checks resumed loss/gradient norm. The block-FP8 case requires `torchao==0.17.0` and the optional `experiments/torchao-npu` package (install with `pip install ./experiments/torchao-npu` from the repository root); it runs real forward/backward/optimizer updates with a quantized frozen base and trainable floating-point adapters. Quantization unit tests follow the repository convention and skip when the optional package is unavailable; explicitly running this NPU case requires the package.

On Ascend 950 devices with the CANN 9.2.0 release runtime (2026-09-09 packages) and block-FP8 support, run the registered quantized case through the same runner:

```bash
python -m tests.integration_tests.run_tests ./test_reports/lora-fp8 \
  --test_suite deepseek_v4_quantized --test_name dsv4_lora_block_fp8_ep2_fsdp2 --ngpu 2
```

The A3 CI `models` suite runs floating-point LoRA training and PEFT export. Distributed DCP resume stays in the dedicated checkpoint suite because the A3 CI runtime fails to load `libscatter_aicpu_kernel.so` during checkpoint planning. Run resume on a runtime that supports this distributed checkpoint path:

```bash
python -m tests.integration_tests.run_tests ./test_reports/lora-resume \
  --test_suite deepseek_v4_checkpoint --test_name dsv4_lora_resume_ep2_fsdp2 --ngpu 2
```

The quantized suite requires the block-FP8 hardware/runtime.

## Engram HF 专项验证（手动）

在支持 Engram MXFP8 Host 通信的超节点上激活 CANN、torch extension 和训练环境，准备 V4.1 tokenizer assets 后执行：

```bash
HF_ASSETS_PATH=/path/to/v41-tokenizer \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
python -m tests.integration_tests.run_tests /tmp/engram-hf-output \
  --test_suite deepseek_v4_1_engram_hf --ngpu 4 --no-parallel
```

输出目录使用新目录。该用例为 V4.1 debug text、四卡 EP4/FSDP4、eager、seq512、GBS4，启用 Engram MXFP8 override，关闭普通模型 FP8 和 optimizer CPU offload。使用仓内 C4 数据及自动生成的微型 HF fixture，不需要正式模型权重。

先训练两步并保存同一模型的原生 DCP 和 FP32 HF 权重，再各自通过真实 CheckpointManager 初始化模型，使用相同的新优化器和数据种子训练三步。每个 rank 校验加载后的 Engram 参数及 MXFP8 缓存/scale 有效行字节，随后对比 TensorBoard loss/grad_norm。失败会使测试返回非零，成功输出 `HF_ROUNDTRIP PASS`。不读取固定 golden，不验证优化器状态续训，也不覆盖正式整包非 Engram 量化权重的导入。
