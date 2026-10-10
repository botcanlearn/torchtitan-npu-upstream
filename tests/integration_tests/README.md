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

## Nightly All Models（Lite Actions 调度）

正式测试定义位于 `tests/integration_tests/nightly_all_models_test/`，只维护 `build_test_list()` 返回的 `OverrideDefinitions`。Lite Actions 专属适配器在 `tests/integration_tests/tools/lite_actions/entrypoint.py`，GitHub Artifact / Waiter 位于 `.github/scripts/lite_actions/`。仓内不再使用 `ci_registry.json`；测试本身定义 `test_name/ngpu/nnodes/override_args/expected_steps`，`OverrideDefinitions.env_vars` 持有 CANN/HF/Checkpoint 路径和本用例的模型环境。Lite Actions 只管理 SSH/HCCL **物理拓扑**、设备分配、任务执行及结果回传，不保留会随模型或版本变化的路径。

8P 有两条**不同优化器训练路径**，不是仅用训练步数区分的重复 smoke：

| Case ID | Optimizer | 脚本与参数 | TensorBoard |
| --- | --- | --- | --- |
| `dsv4_flash_a3_8p_example` | Muon（共享脚本默认 `swap_optimizer`） | Flash 8P 原始 recipe + `--training.steps 5 --compile.no-enable` | 1–5 |
| `dsv4_flash_a3_8p_adamw` | AdamW（`OPTIMIZER_OVERRIDES=''` 关闭 Muon swap） | 同一 recipe + `--training.steps 5 --compile.no-enable --optimizer.name AdamW` | 1–5 |

AdamW 的空 `OPTIMIZER_OVERRIDES` 是**已有共享 Shell 开关**，不会覆盖、复制或删减源 recipe 自己维护的 NPU 算子 imports；Muon case 不传此环境变量。原始 example 的 Muon、Inductor、100 steps 默认保持不变。单机训练继续使用 TorchTitan 现有 `run_tests`，不新增专属调度框架。

16P 复用 `examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh`：`OverrideDefinitions.override_args` 声明 EP16、DP shard16、GBS128、AdamW、Eager、5 steps、关 checkpoint、MoE force-load-balance。通过对应 `env_vars` 同时承载 CANN/HF/Checkpoint 路径、非默认 `CONFIG`、`OPTIMIZER_OVERRIDES=''` 以及 MASTER_PORT/HCCL_IF_BASE_PORT；完整 NPU imports 列表仍由源脚本产生。多机校验直接使用对应 case 的 `expected_steps`。多机实际超时由 Lite Actions `config/pipelines.json` 控制，case 不声明另一份无效超时。

### 模型环境归属（OverrideDefinitions.env_vars）

8P 两条 `OverrideDefinitions` 分别直接声明自身的 `env_vars`。例如 Muon case：

```python
OverrideDefinitions(
    test_name="dsv4_flash_a3_8p_example",
    # ...
    env_vars={
        "ASCEND_SET_ENV_PATH": "/mnt/share/Ascend/20260805101249091/ascend-toolkit/latest/set_env.sh",
        "HF_ASSETS_PATH": "/mnt/share/models/DeepSeek-V4-Flash-bf16",
        "CKPT_INIT_LOAD_PATH": "/mnt/share/dsv4_ckpt_8rank",
    },
)
```

AdamW case 直接定义同样三项路径，并额外包含 `"OPTIMIZER_OVERRIDES": ""`。不再通过模块级共享常量间接覆盖，每个 testcase 的环境配置独立可读。`runner.py` 不再重复校验这些用例级资产路径，CANN/HF/Checkpoint 的实际加载由执行时相应工具负责。

`ASCEND_SET_ENV_PATH` 不再从 Lite Actions 配置读取。调度机对每项固定 SHA 的 `build_test_list()` 进行可信发现，将对应 `env_vars` 传给通用 SSH Runner；在执行机先 source 所选 CANN 路径，再 export 用例环境。生成输出目录 `CKPT_SAVE_LOAD_PATH` 是 Runner 的通用运行时职责，而非需要维护的模型资产配置。物理 SSH 地址/HCCL 小网 IP 和 NPU IDs 仍由 Lite Actions 拓扑管理。**A5 资产目录和 CANN 安装路径尚未核实，在其用例中保留空值且通道禁用；启用前必须修改模型仓定义。**

### GitHub Actions 输入与多用例结果

手工触发固定名 `a3-8p-lite-actions.yml` 时，`workflow_dispatch.inputs.test_cases` 可选择单个 test ID：

```json
[{"test_id":"dsv4_flash_a3_8p_adamw"}]
```

或一个受信任 suite，运行 8P Muon 和 AdamW 两项：

```json
[{"suite":"a3_8p_tests"}]
```

```bash
gh workflow run a3-8p-lite-actions.yml --ref refactor/unified-a3-a5-ci \
  -f 'test_cases=[{"suite":"a3_8p_tests"}]'
```

GitHub 输入**仅选择 case/suite**，不接受 `params`、`STEPS`、任意模块路径或 Shell 命令。Artifact 与 `run_id/attempt/commit SHA` 绑定；调度器在执行前从该 SHA 的源码读取 `build_test_list()`，检查唯一性、disabled、物理拓扑和展开后的测试总数（A3 ≤ 2；A5 ≤ 1，仍禁用）。同一 Run 持有一份设备资源锁，依次执行并分别存储日志，返回 `PASS/FAIL/NOT_RUN` 的逐 case GitHub Commit Comment；启动后续 case 前必须确认选定 NPU 已空闲，否则停止并标记 `NOT_RUN`。

唯一受支持的手工适配器是以下固定入口（先在设备上准备 CANN、NPU、HF assets 和网络环境，并确认资源空闲）：

```bash
python3 -m tests.integration_tests.tools.lite_actions.entrypoint inspect --suite a3_8p_tests
python3 -m tests.integration_tests.tools.lite_actions.entrypoint launch \
  --test-id dsv4_flash_a3_8p_adamw --output-dir ./test_reports/adamw
```

双机训练分别使用该入口的 `launch` / `verify`，要求两节点一致的模型 SHA、`NODE_IPS`、NPU 分配和模型资产。**请注意：** 历史 8P 双 case PASS（5/3 steps）并不等于这里新引入的 Muon/AdamW 双路径已经实机通过；过去 16P Eager PASS 也不等于本次优化器开关改造已回归。A5 64P 继续禁用、未实机验证。

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
