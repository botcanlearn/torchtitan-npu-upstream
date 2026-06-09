---
paths:
  - "torchtitan_npu/models/**/*.py"
  - "torchtitan_npu/models/**/train_configs/*.toml"
---

# 模型代码规则

适用范围：`torchtitan_npu/models/`

## 继承上游规则

### 保持模型精简可读

- 模型文件只包含模型架构，不含训练基础设施。
- 权重初始化放在配置或专用 init 函数中，不散布在 `Module.__init__` 中。
- 模型变更后确保原始 checkpoint 仍能正确加载。

### 审查模型目录

修改共享组件（attention、normalization、MoE routing）时，检查并更新 `torchtitan_npu/models/` 下所有受影响的模型目录。不要在某个模型中留下过时模式；若某个上游模型本仓尚未适配，在审查结论中明确说明不适用。

### 跨模型统一

- 不要给每个模型创建功能相同的独立 wrapper。尽量只有一个通用 wrapper 供所有模型共享。
- 上游已有通用实现时，优先复用 `torchtitan.models.common`。
- 被多个模型复用的组件应放在共享位置，而不是复制到每个模型目录，新增跨模型 attention、feed-forward、decoder、MoE、rope、normalization 等组件时，优先建立 `torchtitan_npu/models/common/` 或贡献到上游 common 目录。
- 新增 rotary embedding、MoE router 等组件前，检查已有实现是否已支持该用例。

### 标准模型目录结构

每个模型目录遵循一致的模式：

- `config_registry.py` — 注册模型配置（size、超参）
- `parallelize.py` — 定义模型的并行化策略
- `model.py` / `moe.py` / `state_dict_adapter.py` 等 — 模型定义、专用层或 checkpoint 适配

已有轻量适配目录（如只提供 `state_dict_adapter.py` 或只注册配置）不必强行补齐所有文件；新增完整模型时再遵循上述结构。

### 不要过度特化

仅一个模型需要的功能在该模型目录实现。不要为单个模型的需求修改共享基础设施或基类。

### Forward 中的控制流

保持控制流（路由决策、条件逻辑）在 `forward` 方法中。不要把重要分支逻辑埋在 helper 方法中。

## 插件仓扩展规则

### 模型注入机制

- 插件仓通过 `_inject_module()` 将新模型注入 `sys.modules`，使上游代码能透明发现。
- 新增模型时需在 `__init__.py` 的 `_apply_patches()` 中同时注册到 `_supported_models` 并调用 `_inject_module()`。

### NPU 模型适配

- NPU 特有的模型修改（如算子替换、内存优化）应通过 converter 或 patch 实现，不要直接 fork 上游模型代码。
- 如必须 fork 模型代码，在文件头注释中说明 fork 原因和对应的上游文件位置。
- 如果模型依赖上游 `torchtitan.experiments.*`（当前如 VLM），把实验性依赖限制在对应模型适配层内，不要为实验需求修改通用 patch、converter、distributed helper 或训练入口。
