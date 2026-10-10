# 集成测试基础设施

本目录遵循 Torchtitan 的 `tests/integration_tests` 布局，负责维护集成测试定义、测试入口以及可选的 loss 精确比较。基础架构代码由
torchtitan 迁移而来。

当前注册 DeepSeek-V4 与 DeepSeek-V3.2 模型的集成用例。

## 测试矩阵

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


**A3-8p-CI** 使用独立手工触发的 GitHub Actions。调度机通过 SSH 将精确
Commit SHA 的源码压缩包发送至执行机；源码、全量运行日志、TensorBoard 与
`exit_code.txt` 仅在 `/mnt/share/ci_tests/<北京时间>_run-<run-id>_attempt-<n>/`
保存。GitHub 仅收到 PASS/FAIL、退出码及精选日志（成功时筛选含
`tps:` / `elapsed_time_per_step` 的前 20 行；失败或超时时默认取最后 20 行，
如末尾已被清理日志覆盖，则优先保留异常文件名/行号/代码片段及最后几行，总计不超过 20 行），
调度机记录 `upload-metrics.json`。如果训练因超时或异常终止、Runner 的
`run.log` 未刷新，执行机改从 Rank 0 的 `structured_logs` 提取最后 20
条事件生成 `tail_20.log`，确保失败的 Actions 日志包含实际训练上下文。

当前 8P Lite Actions Eager smoke 的训练参数由 `OverrideDefinitions.override_args` 提供，而不再通过 example Shell 的 `COMPILE_ENABLE` 或 `STEPS` 环境分支传递。共享 examples 入口保留上游默认的 Inductor 与 100 steps；CI 使用后置 `--compile.no-enable --training.steps <n>`。Eager 冒烟通过不代表 Inductor 通过。

## Lite Actions Nightly All Models：测试定义与调度分离

正式测试定义位于 `tests/integration_tests/nightly_all_models_test/`；保持 `nightly_all_models_test` 这个目录名，并采用 `a3_8p_tests.py`、`a3_16p_tests.py`、`a5_64p_tests.py` 与共享 `runner.py`。稳定的 Lite Actions 适配入口是 `tests/integration_tests/tools/lite_actions/entrypoint.py`；可删除的 GitHub Artifact 桥接和 Waiter 归入 `.github/scripts/lite_actions/`，Workflow 名称带 `-lite-actions.yml`，不冒充直接在 GitHub 执行 NPU 训练。

**唯一测试定义源：** 每个模块的 `build_test_list()` 返回标准 `OverrideDefinitions`；其 `test_name`、`ngpu`、`nnodes`（默认 1）、`override_args`、`expected_steps` 等定义由模型源码维护。`ci_registry.json` 已删除；环境部署参数如 CANN、HF 路径、Checkpoint 挂载、SSH/HCCL IP 放在 Lite Actions 的 `config/pipelines.json`/`pools.json`，不放回 testcase。多机阶段仅由通用 Runner 执行已授权测试的 launch/verify，不复制完整训练 recipe。

**工作流选择协议：** `workflow_dispatch.inputs.test_cases` 为有界 JSON 数组，支持单项 test ID 或可信 suite 名称：

```json
[{"test_id":"dsv4_flash_a3_8p_example","params":{}}]
```

```json
[{"suite":"a3_8p_tests","params":{}}]
```

第二种会通过固定 Commit SHA 的真实 `a3_8p_tests.build_test_list()` 展开为 `dsv4_flash_a3_8p_example`（默认 5 steps）和 `dsv4_flash_a3_8p_multicase`（3 steps）。用户不能提交 Python 模块路径、任意 Shell 片段或资源拓扑；`params` 当前仅支持有界 `STEPS` 字符串，由 case builder 转为 `--training.steps` CLI。此处不依赖 Shell 环境开关改变训练语义。

```bash
gh workflow run a3-8p-lite-actions.yml --ref refactor/unified-a3-a5-ci \
  -f 'test_cases=[{"suite":"a3_8p_tests","params":{}}]'
```

调度机从同一个 GitHub Run 的 Artifact 读取绑定 `run_id/attempt/SHA` 的输入，使用目标 SHA 的可信 `build_test_list()` 校验唯一 ID、disabled 状态、`nnodes/ngpu` 是否匹配物理分配，并**按展开后的实际数量**检查通道预算（A3 最大 2 项，A5 最大 1 项且目前禁用）。这个可信代码发现过程需要既有 actor/branch/SHA 准入，**仅凭 SHA 并不意味着代码无害**。

同一 Run 持有资源锁逐个执行用例：所有测试都有独立日志与结果。`PASS/FAIL/NOT_RUN` 通过原有 Lite Actions Commit Comment 协议一次性汇总，GitHub Waiter 会打印完整短表格；遇失败仅在确认上一项进程及 NPU 释放后才继续执行。释放状态不明就停止后续用例、标记 `NOT_RUN`、整个 Action 判定 Failure。

16P Eager smoke 的 EP16/DP16/GBS128/AdamW/compile off/checkpoint off/force-load-balance 均在 `OverrideDefinitions.override_args` 中；共享 Flash recipe 恢复原样。复用 Tyro 后置 CLI 同名覆盖语义，16P `--override.imports` 会替换默认的包含 Muon swap 的列表以避免不相容优化器。多机 TensorBoard 校验从用例自身 `expected_steps` 读取，和实际训练 CLI 使用同一个步数。

**验证边界：** 下文的历史 A3 8P/16P 成功只覆盖当时的提交；本次 suite 重构需新的 Actions Run 成功后才能宣称回归。A5 64P 仍禁用、未实机验收。

## A3 8P DeepSeek-V4 Flash Eager 测试（正式 testcase）

`tests/integration_tests/nightly_all_models_test/a3_8p_tests.py` 直接复用仓内 `run_tests.py` 与 `examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh`。当前 8P 模块内有两个真实测试定义：`dsv4_flash_a3_8p_example`（5 steps）与 `dsv4_flash_a3_8p_multicase`（3 steps）。二者均为单节点 8 张 A3 NPU、Eager smoke，使用 `--compile.no-enable` 覆盖 example 的默认 `--compile.enable`；TensorBoard 校验随各自 `expected_steps` 执行。

在已准备好 CANN、torch_npu、TorchTitan 和 HF assets 的独立环境中，仅手工运行某个 case（注意先验证这台机器的 NPU 空闲）：

```bash
HF_ASSETS_PATH=/path/to/DeepSeekV4_tokenizer \
python3 -m tests.integration_tests.nightly_all_models_test.a3_8p_tests \
  dsv4_flash_a3_8p_example ./test_reports/dsv4_flash_a3_8p
```

指定 `LITE_TEST_STEPS=10` 可修改第一项的 smoke 步数，具体 `--training.steps` 与 `expected_steps` 同源；第二项仍保持独立的 3 steps。不要将 `STEPS=5` 或 `COMPILE_ENABLE=0` 当作此版本的训练参数协议。正式多用例验收请通过 `a3-8p-lite-actions.yml` 输入 `[{"suite":"a3_8p_tests","params":{}}]` 发起；总 Runner 会顺序执行、隔离日志、汇总 `PASS/FAIL/NOT_RUN`。原始 example 直接运行仍保留其既定默认值，Inductor / Muon 不包含在本次 Eager 验收范围内。

## 并行调度

runner 迁移自 torchtitan 的 GPUPool 机制：默认将用例并发打包到固定的 NPU 池上，
每个用例通过 `ASCEND_RT_VISIBLE_DEVICES` 绑定到互不相交的物理 NPU 子集，
任一时刻在用 NPU 数量不超过设备池大小。设备池从真实可见性构造：若运行环境已通过
`ASCEND_RT_VISIBLE_DEVICES` 限定可用 NPU 子集（如 CI 按任务分配设备），池从该
子集构造并对超出的 `--ngpu` 硬报错；否则用 `torch.npu.device_count()` 枚举运行时
实际暴露的物理 ID，`--ngpu` 超出实际设备数时告警并截断——绝不按 `range(--ngpu)`
伪造 ID（不存在的 ID 会让子进程在 CANN `GetVisibleDevices` 阶段即失败，torchtitan
设备探测退回 "cuda" 后以 `torch._C._cuda_setDevice` AttributeError 崩溃）。池小于
某用例需求时该用例被显式 skip 而非在 `acquire()` 中死锁。用例按 `ngpu` 从大到小
提交以减少队头阻塞；并行结束后 runner 会输出两行调度遥测：池利用率
（`[parallel] pool: window/utilization/busy histogram/allocations`）与
用例重叠（`[parallel] overlap: sequential vs window、节省时长、并发度直方图`），
统计窗口均为首次分配到最后一次释放，可直接用于核验 CI canary 的打包与重叠效果。
各用例的输出被整体缓存，结束后以带 `[case 名]` 前缀的连续块输出，避免多用例
日志交错。如需强制串行执行，传入 `--no-parallel`。用例可通过
`OverrideDefinitions.timeout` 设置超时；超时后 runner 会向子进程所在进程组先发
`SIGTERM`、宽限期后再发 `SIGKILL`，确保 `torchrun` 及各 rank 子进程全部退出，不会
留下占用 NPU 的孤儿进程（超时按失败处理并输出已捕获日志）。

- 同一台机器上并发运行两个 2 卡 case 时必须为每个 run 设置不同的
  `HCCL_NPU_SOCKET_PORT_RANGE`（例如 `62000-62020`）；否则后启动的 run 会在 HCCL
  建链时报 `Communication_Error_Bind_IP_Port`（一次并发验证即在同一端口上冲突）。

调度器本身不设独立单元测试：其正确性（设备不重叠、失败/超时释放、并发打包、
golden loss 等价）由集成测试自身的 canary 运行直接验证。

### LoRA training and resume

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


## A3 16P DeepSeek-V4 Flash（双机 Eager）

16P 复用 `examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh`，不维护第三份完整 Flash recipe。`a3_16p_tests.py` 将运行配置作为环境传递：每机 8 卡、两机、EP16、DP shard16、GBS128、5 steps、AdamW、关闭 compile 和 checkpoint、MoE force-load-balance。Muon/Inductor **不在此 Eager smoke 覆盖范围内**。模型仓 Unit Tooling Tests 会通过模拟 `run_train_multinodes.sh` 检查最终展开的参数，并确保共享 Flash 默认 Muon/Inductor 配置保持不变。

GitHub 已有历史 Eager 验收：[A3 16P Run 37942444656](https://github.com/depeng1994/torchtitan-npu/actions/runs/37942444656)（双节点 5 steps、TensorBoard、GitHub Success）。其成功仅证明旧提交，重构后必须重新完成实机 Actions 回归才能引用为新版本 PASS。

启动/停止统一由 Lite Actions 主机的可信 Workflow → Agent → SSH Runner 完成；不要直接运行不存在的 `--stage-only`、`--run-dir` 等旧参数。独立环境手工排障可按受信任 `lite_actions.tools.entrypoint` 运行 `launch/verify`，两节点需要相同的注册表和准确的 `NODE_IPS`，且不可占用其他训练作业。

A5 64P 目前仅完成静态拓扑和命令展开检查，**未实机验证**。
