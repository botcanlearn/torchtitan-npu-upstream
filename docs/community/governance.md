# 社区治理

## 使命

`torchtitan-npu` 致力于为昇腾 NPU 提供面向 `torchtitan` 的训练适配与优化能力，
让 PyTorch native 的大模型训练能够在昇腾平台上稳定、高效、可观测地运行。

项目重点包括：

- 保持与上游 `torchtitan` 的训练架构、配置系统和分布式能力协同演进。
- 提供 NPU 亲和优化，包括融合算子、图优化、图下沉、算子自动融合和显存优化。
- 支持大模型训练场景中的并行策略、模型适配、精度验证、性能分析和调试维测。
- 通过开放协作沉淀可复用的训练经验，服务 CANN 与 PyTorch 生态开发者。

## 基本原则

- **开放协作**：欢迎 Issue、PR、文档、测试、实验报告和场景反馈等多种形式的贡献。
- **代码为准**：技术结论、兼容性说明和性能数据应有代码、配置、日志或实验结果支撑。
- **上游友好**：与 NPU 平台无关的通用能力优先回馈上游 `torchtitan`，本仓聚焦昇腾适配和优化。
- **质量优先**：影响训练正确性、性能或兼容性的改动必须提供必要验证。
- **可追溯**：版本兼容、上游基线、重大决策和治理角色变化应通过文档、Issue、PR 或 SIG 会议记录留痕。

## 治理结构

`torchtitan-npu` 是 CANN 组织下的开源项目，日常技术治理与社区协作依托
GitCode 仓库和 CANN `framework-adapter` SIG 展开。更高层级的社区组织与治理
遵循 CANN 社区统一安排。

本项目的正式角色定义、晋升标准、投票流程和非活跃退出机制遵循
[CANN 社区角色定义及晋升机制](https://gitcode.com/cann/community/blob/master/governance/role-definition-and-promotion-mechanism.md)。
本文仅结合 `torchtitan-npu` 项目说明各角色在本仓中的协作边界。

### User

User 指使用 `torchtitan-npu` 的用户。User 可以通过 Issue、讨论、SIG 会议或
PR 反馈安装、训练、精度、性能、文档和易用性问题，也可以提出新模型、新算子、
新并行策略或调试能力需求。

### Contributor

Contributor 指以任何形式参与项目的人，包括但不限于：

- 提交文档、测试、代码、配置或性能数据。
- 参与 PR review、SIG 讨论、版本验证或问题定位。

Contributor 不要求拥有仓库写权限。一次有效贡献即可成为 Contributor。

### Committer

Committer 是具备特定代码仓库写权限的项目成员。`torchtitan-npu` 的 Committer
权限和职责按本仓维度申请与管理。典型职责包括：

- 维护关键代码路径和文档质量。
- 合入已满足质量要求的 PR。
- 协助处理紧急缺陷、回归问题和版本发布前的阻塞项。
- 推动 CI、测试、文档和开发流程持续改进。
- 处理社区 Issue，帮助 Contributors 提升代码开发和问题定位能力。

Committer 应持续展现高质量贡献、负责任的 review 和对项目方向的理解。

### Maintainer

Maintainer 负责特定 SIG 的运营和维护。对于 `torchtitan-npu`，Maintainer
主要通过 CANN `framework-adapter` SIG 协调项目方向和社区健康度。典型职责包括：

- 制定或维护 Roadmap、版本策略、上游同步策略和关键技术方向。
- 对重大架构调整、兼容性变化、发布节奏和治理事项做最终协调。
- 组织或参与 SIG 例会、版本复盘、技术方案评审和社区沟通。
- 吸纳并发展 Committer，推动项目孵化、推广和长期维护。
- 按 CANN 社区要求向相关委员会同步 SIG 运营进展。

Maintainer 应对 `torchtitan`、`torchtitan-npu`、昇腾软件栈和大模型训练流程有深入理解，
并长期参与项目建设。

### PMC 与 TSC

项目管理委员会（PMC）和技术指导委员会（TSC）的职责、产生方式和退出机制遵循
CANN 社区统一治理文件。`torchtitan-npu` 的重大社区治理事项应按需提交到
`framework-adapter` SIG、PMC 或 TSC 流程中。

## 决策机制

项目优先采用共识决策：

1. 一般改动通过 PR review 达成共识后合入。
2. 影响架构、兼容性、性能基线、上游同步或发布策略的改动，应在 Issue、PR
   或 SIG 会议中充分讨论。
3. 若存在分歧，由相关模块 Maintainer 组织讨论并给出处理意见。
4. 对项目方向有重大影响的事项，应同步到 CANN `framework-adapter` SIG 或
   CANN 社区治理流程中。

## 晋升与退出机制

`torchtitan-npu` 不单独维护一套独立晋升标准。Committer、Maintainer、PMC、
TSC 的晋升、投票、公告、配置更新和非活跃退出流程均以
[CANN 社区角色定义及晋升机制](https://gitcode.com/cann/community/blob/master/governance/role-definition-and-promotion-mechanism.md)
为准。

对于本仓 Committer 申请，应按代码仓库维度准备贡献材料，并通过
`framework-adapter` SIG 流程推进。候选人通常需要证明其对 `torchtitan-npu`
相关代码路径足够熟悉，且具备持续的代码贡献、PR review、问题处理和社区协作记录。

Maintainer 晋升应通过 CANN `framework-adapter` SIG 例会和 CANN 社区统一机制完成。
非活跃成员退出也遵循 CANN 社区统一定义和处理流程。

## 发布与版本治理

`torchtitan-npu` 采用“分支 + commit 基线”的方式与上游 `torchtitan` 对齐。
版本兼容性、分支同步信息和上游基线应以 [版本策略](./versioning_policy.md) 为准。

每次完成上游同步、发布适配或依赖兼容性调整后，应同步更新：

- `docs/community/versioning_policy.md`
- `docs/user-guides/installation.md`
- `README.md`
- 相关特性文档、测试指南或配置示例

## 贡献流程

贡献者应遵循 [贡献指南](../../CONTRIBUTING.md)：

- 从 `master` 创建分支并提交 PR。
- 影响功能的改动应补充或更新测试。
- 提交前通过格式检查和必要测试。
- PR 描述应包含改动动机、实现方案和验证结果。
- 涉及精度或性能的改动，应提供可复现的实验说明。
- 涉及上游 `torchtitan` 同步的改动，应同步更新版本策略。

## 行为准则

社区讨论应保持专业、尊重和建设性。不同意见应围绕事实、代码、实验数据和用户场景展开。
对恶意攻击、骚扰、泄露敏感信息或破坏社区协作的行为，维护者可根据项目治理流程进行处理。
