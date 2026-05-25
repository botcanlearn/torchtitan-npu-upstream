# 提交者与贡献者（Committers and Contributors）

感谢所有参与 `torchtitan-npu` 的开发者。`torchtitan-npu` 是基于 `torchtitan`
的昇腾全流程大模型训练适配插件，社区贡献共同推动了 NPU 融合算子、图优化、
分布式并行、调试维测、模型适配和文档体系的持续演进。

## Committers

`torchtitan-npu` 的日常维护由 GitCode 仓库 Committers 与 CANN
`framework-adapter` SIG 协同推进。Committers 负责协助维护代码质量、
评审社区贡献、推动版本适配，并按项目治理流程合入已满足质量要求的 PR。

| GitCode ID | GitCode 主页 |
| --- | --- |
| depeng1994 | [https://gitcode.com/depeng1994](https://gitcode.com/depeng1994) |
| zhong_lin | [https://gitcode.com/zhong_lin](https://gitcode.com/zhong_lin) |
| lrwei0709 | [https://gitcode.com/lrwei0709](https://gitcode.com/lrwei0709) |
| zzyyjj012 | [https://gitcode.com/zzyyjj012](https://gitcode.com/zzyyjj012) |
| xuyujun | [https://gitcode.com/xuyujun](https://gitcode.com/xuyujun) |
| panchao-gitcode | [https://gitcode.com/panchao-gitcode](https://gitcode.com/panchao-gitcode) |
| zhaowei1936 | [https://gitcode.com/zhaowei1936](https://gitcode.com/zhaowei1936) |
| zhanghz1 | [https://gitcode.com/zhanghz1](https://gitcode.com/zhanghz1) |

Committer 名单、仓库权限和职责边界以 GitCode 仓库权限配置、CANN 社区治理安排
以及 SIG 会议纪要为准。

相关入口：

- 项目仓库：[cann/torchtitan-npu](https://gitcode.com/cann/torchtitan-npu)
- 贡献指南：[CONTRIBUTING.md](../../CONTRIBUTING.md)
- SIG 页面：[framework-adapter](https://gitcode.com/cann/community/tree/master/CANN/sigs/framework-adapter)
- SIG 例会：[sig-framework-adapter](https://meeting.osinfra.cn/cann?sig=sig-framework-adapter)

如果需要新增或更新 Committer 信息，请通过 PR 修改本文档，并在 PR 中说明依据，
例如仓库权限调整、SIG 决议或持续贡献记录。

## Contributors

以下列表基于 `master` 分支 git author 历史生成，统计范围截至
`29718c1`。该列表仅用于感谢和记录贡献，不代表仓库权限或治理角色。

| 序号 | 贡献者 | 首次贡献 | 最近贡献 | 提交数 |
| --- | --- | --- | --- | --- |
| 1 | mystri | 2026/01/29 | 2026/05/09 | 25 |
| 2 | CjianForBetter | 2026/03/02 | 2026/05/09 | 14 |
| 3 | hitwdy | 2026/02/28 | 2026/05/20 | 13 |
| 4 | xuyujun | 2026/01/29 | 2026/05/20 | 11 |
| 5 | caojingyi | 2026/01/30 | 2026/04/13 | 9 |
| 6 | zhanghz1 | 2026/01/27 | 2026/03/11 | 9 |
| 7 | zhangwei1177 | 2026/03/11 | 2026/05/09 | 9 |
| 8 | xubin787 | 2026/01/26 | 2026/03/20 | 8 |
| 9 | qianbi1999 | 2026/02/04 | 2026/03/30 | 6 |
| 10 | depeng1994 | 2026/01/27 | 2026/04/24 | 4 |
| 11 | lrwei0709 | 2026/02/11 | 2026/03/31 | 4 |
| 12 | zhaiyukun | 2026/03/18 | 2026/05/15 | 4 |
| 13 | zzyyjj012 | 2026/04/01 | 2026/05/19 | 4 |
| 14 | liuyuanchen1 | 2026/04/25 | 2026/05/07 | 3 |
| 15 | MissingPompeii | 2026/03/26 | 2026/05/09 | 2 |
| 16 | cann-robot | 2026/01/21 | 2026/01/30 | 2 |
| 17 | zhangjianshe | 2026/04/01 | 2026/04/10 | 2 |
| 18 | Yong-An | 2026/04/30 | 2026/04/30 | 1 |
| 19 | aoyoukeji | 2026/04/29 | 2026/04/29 | 1 |
| 20 | m0_62051609 | 2026/05/18 | 2026/05/18 | 1 |
| 21 | panchao-gitcode | 2026/04/02 | 2026/04/02 | 1 |
| 22 | weijie11 | 2026/03/11 | 2026/03/11 | 1 |
| 23 | weixin_47893573 | 2026/04/30 | 2026/04/30 | 1 |
| 24 | zcklllyao | 2026/05/20 | 2026/05/20 | 1 |

## 如何成为贡献者

任何人都可以通过以下方式参与 `torchtitan-npu`：

- 提交 Issue，报告训练、精度、性能、安装或文档问题。
- 提交 PR，修复缺陷、补充测试、完善文档或适配新的 NPU 训练特性。
- 参与 `framework-adapter` SIG 讨论，反馈真实训练场景中的需求和风险。
- 在 PR 中提供清晰的验证结果，例如单元测试、冒烟测试、loss 曲线、
  吞吐量、显存峰值或端到端训练结果。

请在贡献前阅读 [贡献指南](../../CONTRIBUTING.md)。涉及上游 `torchtitan`
基线同步时，还需要同步维护 [版本策略](./versioning_policy.md)。
