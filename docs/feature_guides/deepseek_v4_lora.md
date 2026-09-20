# DeepSeek-V4 LoRA 微调

在 DSV4 recipe 的 `converters` 中选择 `DeepSeekV4LoRAConverter.Config`，即可启用 dense、batched 和 routed-expert LoRA。recipe 自动选择 LoRA 并行化与 PEFT checkpoint manager；无需额外 enable 开关。rank、alpha 和目标模块在 converter 配置中设置，保存选项使用 `checkpoint.*`；暂不支持量化基座。

完整 Flash 模型使用已有的 `deepseek_v4_flash_lora` recipe（dense/expert rank 16、alpha 32），无需创建额外配置模块。正式入口复用 A3 4K launcher 的节点、数据、并行和编译配置，并开启 checkpoint 保存。当前 LoRA 训练仅支持 AdamW；公共 A3 launcher 默认使用 Muon，因此运行时须显式传入 `--optimizer.name AdamW`：

```sh
NODE_IPS="${NODE_IPS}" \
HF_ASSETS_PATH=/path/to/tokenizer \
CKPT_INIT_LOAD_PATH=/path/to/bf16-base-weights \
CKPT_SAVE_LOAD_PATH=/path/to/run/checkpoint \
bash examples/deepseek_v4/deepseek_v4_flash_lora_4k_a3.sh \
  --optimizer.name AdamW \
  --training.steps 500 --checkpoint.interval 100
```

节点和数据环境变量说明见 [训练示例](../../examples/deepseek_v4/readme.md)。命令末尾的 CLI 参数可覆盖 launcher 默认值。自定义 recipe 可继续在 `deepseek_v4_flash(converters=[...])` 中配置 `DeepSeekV4LoRAConverter.Config`。

模型并行化后、optimizer 创建前冻结非 LoRA 参数，并关闭 MoE load-balancing hook，保留基座 routing bias。部分 adapter 训练时，在该 parallelize callback 返回后冻结相应 adapter（例如全部 `lora_a`），仅训练剩余参数。

`checkpoint.periodic_save_adapter_only=True` 时，周期 checkpoint 使用 native DCP，保存 adapter 和 routing buffer，不重复保存冻结的基座。启用 `checkpoint.save_training_state=True` 后，还会保存 optimizer、scheduler、dataloader 和 step；上述 LoRA launcher 已启用此选项。恢复时保持相同基座、训练总步数和运行目录；用 `--checkpoint.load-step 10` 指定恢复点。

最后一步导出 PEFT 的 `adapter_model.safetensors` 和 `adapter_config.json`，仅包含 adapter 权重。训练续跑使用周期 DCP；若最后一步也需保存训练状态，设置 `--checkpoint.no-last-save-in-peft --checkpoint.no-last-save-model-only`。启用 expert adapter 时，PEFT 导出要求 dense 和 expert 的 rank 相同。启用 checkpoint 后，MTP adapter、rank 不兼容和不支持的 PEFT 目标在 checkpoint 初始化时拒绝，而不是训练结束后才失败。`last_save_in_hf=True` 不合并 LoRA 增量，因此本 checkpoint manager 禁止该选项；请使用 PEFT 或 native DCP。

## 加载 PEFT adapter

导出使用 `target_parameters`，保留 batched projection 的分组前向计算，并将 routed expert 的 gate/up adapter 转成融合参数格式。加载时显式指定与训练相同的基座：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("/path/to/hf-base")
model = PeftModel.from_pretrained(base, "/path/to/run/checkpoint/step-100")
```

基座路径须为 Transformers 可加载的模型目录；训练用 native DCP 路径不能直接用于此处。PEFT 导出不支持 MTP adapter；训练 MTP adapter 时使用 native checkpoint，或设置 `include_mtp=False` 后导出 PEFT。

## 验证

CPU 消费者测试在实际 Transformers 模型上加载 dense、batched 和 routed-expert adapter，并与直接合并权重的模型比较 logits：

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/unit_tests/models/deepseek_v4/test_lora_state_dict_adapter.py -k peft_export_loads
```

该测试需要上述可选依赖。NPU 用例复用现有 integration runner，在 EP2/FSDP2 上执行 2 步 A/B 训练，检查冻结基座与 routing bias 不变、adapter 更新。该用例不做中间 checkpoint 保存或恢复；最后一步通过实际 checkpoint manager 导出 PEFT，并检查导出张量的 key、shape、数值及 rank、alpha、target 配置。冻结 A、仅训练 B 由 CPU 单元测试覆盖：

```sh
python -m tests.integration_tests.run_tests /tmp/lora-ab \
  --test_suite models --test_name dsv4_lora_ep2_fsdp2 --ngpu 2 --no-parallel
```

PEFT export requires a local checkpoint output folder; remote output URIs are rejected during initialization.
Set `--checkpoint.peft-base-model-name-or-path` to the Hugging Face base model ID or directory for external loaders.
When omitted, an HF initial checkpoint supplies this metadata; native DCP paths are never exported as the base model.
Otherwise the configured HF assets path supplies the default base metadata.

Final PEFT export reconstructs only LoRA tensors across ranks, one tensor at a time.
Rank 0 writes the CPU adapter tensors to safetensors and the PEFT configuration;
all ranks receive the save result. Native training checkpoints continue to use DCP.
