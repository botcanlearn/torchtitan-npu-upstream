# torchtitan-npu 开发指南

`torchtitan-npu` 是 [torchtitan](https://github.com/pytorch/torchtitan) 的 **Ascend NPU 插件仓**。
本仓不直接修改上游 torchtitan 代码，而是通过 monkey-patch、ModelConverter、模型注入等机制将 NPU 适配能力叠加到上游之上。

## 核心原则

1. **PyTorch 原生训练技术。** torchtitan 核心的训练基础设施和并行代码不依赖非 PyTorch 库。作为插件仓，torchtitan-npu 可使用 torch_npu 等外部库，但应尽可能复用 PyTorch 原生接口。

2. **查明根因再修复。** 不做绷带式修补。在提出方案前理解 *为什么* 出错。如果一个改动看似有效但无法解释原因，需要更深入排查。

3. **复用优于重复。** 新写代码前，检查已有实现是否已覆盖需求。尽量统一跨模型的相似代码路径，不要给每个模型创建独立 wrapper。若上游（torchao、PyTorch）已提供功能，优先使用。

4. **不要将实验泄漏到核心。** 插件仓中如需实验性代码，务必与核心适配逻辑隔离，不要在核心 patch/converter 文件中添加 `if experiment_x:` 分支。

5. **保护已验证的代码路径。** 修改已收敛的代码时务必谨慎。标记可能导致现有用户代码或 checkpoint 静默失效的改动。存疑时主动询问。

6. **审计所有调用点。** 修改共享代码（公共模型组件、配置字段、分布式工具）时，检查并更新所有调用点。这包括所有模型变体如 llama3、llama4、qwen3、deepseek_v3、deepseek_v32 等。

## 插件仓专属原则

1. **绝不修改上游代码。** torchtitan-npu 不直接修改 torchtitan 源码。所有适配通过以下机制实现：
   - **Patches**（`torchtitan_npu/patches/`）：monkey-patch 上游模块的函数或类
   - **Converters**（`torchtitan_npu/converters/`）：通过 ModelConverter 注册表替换算子或注入自定义 kernel
   - **模型注入**（`torchtitan_npu/models/`）：补充或覆盖上游模型实现

2. **理解 patch 生效机制。** 所有 patch 在 `torchtitan_npu/__init__.py` 的 `_apply_patches()` 中注册，包导入即生效。新增 patch 必须在此函数中添加对应 import。

3. **Converter 遵循注册表模式。** 自定义算子转换必须通过 `torchtitan_npu/converters/registry.py` 的 `@register_model_converter()` 注册。不要在模型文件中硬编码算子替换逻辑。

4. **上游同步是常态。** 本仓需定期跟踪上游 torchtitan 变更。同步基线信息维护在 `docs/community/versioning_policy.md` 的分支同步表。每次同步后必须更新此表。

5. **Patch 目标随上游变化。** 上游重构后，patch 的目标函数/类可能已不存在或签名已变。每次上游同步必须检查所有 patch 是否仍有效。

## 代码风格（继承上游 torchtitan）

### 命名

- 名称必须 **准确、描述性、反映实际作用域**。不要在生产代码中使用 "toy/test/temp" — 这类上下文放在 docstring 中。
- 遵循上游约定：匹配 torchao 和 PyTorch 的命名。
- 计数使用 `num_` 前缀（如 `num_expert_groups` 而非 `n_expert_groups`）。

### 代码放置

代码放到 **最通用的适用位置**：

| 目录 | 职责 |
| --- | --- |
| `torchtitan_npu/patches/` | 对上游 torchtitan、PyTorch、torch_npu 等模块的 monkey-patch，按 patch 目标分子目录 |
| `torchtitan_npu/converters/` | 算子转换器注册表与自定义 kernel，通过 registry 机制注入 |
| `torchtitan_npu/models/` | 模型实现（覆盖或扩展torchtitan），含并行化策略与训练配置 |
| `torchtitan_npu/distributed/` | NPU 专属分布式工具 |
| `torchtitan_npu/config/` | NPU 自定义配置扩展 |
| `torchtitan_npu/tools/` | 训练辅助工具（flight_recorder、profiling 等） |
| `torchtitan_npu/train.py` | 训练流程 patch（如模型专属训练逻辑） |
| `torchtitan_npu/entry.py` | 训练入口点 |

不要把模型无关的功能放在模型特定文件中。

### 断言与错误处理

- **`ValueError`** 用于用户可见的错误（配置错误、无效输入）。
- **`assert`** 仅用于表示程序错误的内部不变量。
- 分布式代码中显式验证 mesh 维度、tensor placement 和配置值 — 不要假设 1D mesh 或特定 placement。
- 代码路径静默跳过用户配置时，**发出 warning**。

### 参数与配置

- 重要参数放前面，次要参数放后面。
- 首个位置参数之后优先使用 keyword-only 参数。
- 必需配置字段不要用 `None` 默认值。
- `dataclasses.replace()` 是浅拷贝：嵌套 dataclass 和 list/dict 字段共享引用。需要深拷贝时显式处理。

### 注释与文档

- 仅为真正不明显的内容添加注释：维度语义、并行梯度 placement、workaround 存在的原因。
- 使用 TODO 注释标记已知限制并附简要说明。
- 描述放在 docstring 中，不要放在名称里。
- 注释使用英文，文档优先使用中文。

## 标准开发 Pipeline

### 1. 获取上下文

- 先确认任务涉及的目录、模型、并行策略和是否影响训练数值。
- 修改 `torchtitan_npu/` 下代码时，按路径加载 `.agents/rules/` 中的专项规则；如果同一改动命中多个领域，规则全部适用。
- 涉及上游同步时使用 `torchtitan-sync` skill。

### 2. 实施修改

- 保持改动最小，只改完成目标所需的文件。
- 复用现有 patch、converter、model injection、distributed helper 和 config 模式。
- 新增或修改 patch/converter/model/config 后，同步检查注册入口和所有调用点。
- 对数值、分布式、checkpoint、模型加载路径保持保守；存疑时先给出风险和验证方案。

### 3. Codecheck

step1: 运行pre-commit

```bash
pip install -r requirements.txt -r requirements_dev.txt
# 快速迭代，可先只检查改动文件
pre-commit run --files <changed files>
# 提交或发 PR 前必须通过
pre-commit run --all-files
```
step2: 代码审查

代码审查或定位 codecheck 失败时使用 `torchtitan-npu-code-reviewer` skill。

### 4. 测试与数值验证

- Python 逻辑改动至少运行相关单元测试；影响共享逻辑时扩大到 `pytest tests/ -x`。
- 涉及分布式、NPU kernel、converter、patch 或模型训练行为时，补充对应 NPU 冒烟/集成测试。
- 数值验证：
  - 非计算性改动（重构、activation checkpointing 调整等）必须保证修改前后 **loss 完全一致**；计算性改动需在代表性数据集（如 C4）上展示 loss 收敛。
  - 对齐验证须加载同一 checkpoint 并固定 NPU 随机性，相同并行策略下两次运行的 loss 和 grad_norm 应一致；**禁止** 使用 `--debug.deterministic_warn_only`。
  - 优先用 `premerge-accuracy-check` skill 生成 loss/grad_norm 对比报告，仅对已有日志作图可用 `training-log-visualization`；证明 bit-wise 一致需保留更高精度来源，stdout 的 5 位有效数字不能作为唯一依据。

### 5. PR 与流水线

- **提 PR 前必须调用 `torchtitan-npu-code-reviewer` skill 审查本次改动**，建议修复其报告的 S1/S2 问题后再创建 PR。
- PR 描述解释“为什么”而非只是“做了什么”；非 trivial 改动附 loss 对比曲线，模型变更说明 checkpoint 兼容性。
- 用 `gitcode-pr` 创建/推送 PR、读取改动与评论，并严格按 `.gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md` 填写：标题用英文类型标签 `[type] 描述`，`类型` 只勾一个主类型，`Checklist` 只勾真实完成项，`如何测试` 写实际执行的命令或说明未执行原因。
- 用 `gitcode-pipeline` 触发/等待 CI 并拉取失败日志，失败后结合失败类型与专项规则修复或判断 CodeCheck 屏蔽；缺少上述远程 skills 时先用 `default-skills` 安装。
