# 测试设计

## 文档目的
这份文档说明 `torchtitan-npu` 当前的测试分层，帮助开发者判断测试新用例应该放在哪一层。

测试目标：
- 单元测试保持快速、硬件无关
- 冒烟测试聚焦真实执行链路
- 让开发者更容易判断测试该写在哪里

## 测试分层
| 分层 | 目录 | 是否需要 NPU | 适用场景 |
|---|---|---|---|
| 函数级单元测试 | `tests/unit_tests/functions/` | 否 | 纯函数、配置解析、小工具、参数校验 |
| 模块级单元测试 | `tests/unit_tests/modules/` | 否 | wrapper、checkpoint、分布式初始化等模块逻辑 |
| 转换器单元测试 | `tests/unit_tests/converters/` | 否 | converter 注册、替换、映射逻辑 |
| 补丁单元测试 | `tests/unit_tests/patches/` | 否 | patch 激活、接线和小范围行为验证 |
| 特性冒烟测试 | `tests/smoke_tests/features/` | 是 | 真实 NPU 特性链路、融合算子、wrapper 执行链 |
| 模型并行冒烟测试 | `tests/smoke_tests/model_parallel/` | 是 | CP/TP/EP、mesh、DTensor、模型并行场景 |

## 怎么判断放哪层
- 不需要 NPU 就能验证的，优先放单元测试
- 必须依赖真实 NPU 才有意义的，放冒烟测试
- 涉及 mesh、shard、placement、DTensor 的，放模型并行冒烟测试
- 只是小工具或纯转换逻辑的，放单元测试

## 本仓里的 Smoke 是什么
这里的 smoke 不是简单的导入检查，而是对真实执行链路做集成式验证。

`bash .ci/smoke_test.sh` 会先跑 torchtitan-npu 集成 smoke（`run_torchtitan_npu_smoke`，
内部驱动 `integration_test.py`），再跑 `pytest tests/smoke_tests`。


## 新增测试的基本规则
1. 优先验证真实行为，不要只做导入检查。
2. 不要用 placeholder 测试充数。
3. 单元测试必须保持硬件无关。
4. 只有在真实 NPU 上才有意义的行为，才放到 smoke。
5. 如果测试依赖外部产物或特殊运行环境，要写清楚。

## 可读性要求
- 测试名直接表达“行为 + 预期结果”
- setup 尽量简短
- 必要时用空行区分 Arrange、Act、Assert
- 断言要直接体现这个测试在保护什么

避免：
- 很长的测试函数 docstring
- 只是重复测试名的注释
- 没有实际意义的分隔线注释

## 执行入口
- `bash .ci/unit_test.sh`：运行全部单元测试
- `bash .ci/smoke_test.sh`：运行全部冒烟测试

常用定向命令：
- `python tests/smoke_tests/integration_test.py <输出目录> --test_name <name>`：单独跑某个集成用例
- `python3 -m pytest tests/smoke_tests/model_parallel/`：直接跑模型并行 smoke

## 提交前自查
1. 测试是不是放在正确目录？
2. 验证的是不是实际行为，而不是占位路径？
3. 其他开发者能不能很快看懂这条测试？
4. 如果改动影响了测试入口或使用方式，文档有没有同步更新？
