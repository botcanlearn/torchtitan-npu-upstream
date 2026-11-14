# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the submit-pr PR body template validator."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = ROOT / ".agents/skills/submit-pr/scripts/validate_pr_body.py"
TEMPLATE = ROOT / ".gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md"

_SPEC = importlib.util.spec_from_file_location("submit_pr_validate_pr_body", VALIDATOR)
validate_pr_body = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validate_pr_body)

VALID_BODY = """## 描述
说明改动。

## 类型
- [ ] Bug 修复
- [x] 新功能
- [ ] 重构（即不是新增功能，也不是修改bug的代码变动）
- [ ] 构建过程或辅助工具的变动
- [ ] 文档内容更新

## Checklist:
- [x] 我的代码遵循这个项目的代码风格
- [x] 我已经自己测试过我的代码
- [ ] 我已经更新了相应的文档
- [x] 我已经在标题中正确使用了类型标签（例如：`feat`, `fix`, `refactor`, `docs`, `test`）

## 如何测试
执行了测试。

### 精度和性能验证结果
| 项目 | 结果 |
| --- | --- |
| loss | 通过 |

## 其他信息
无。
"""


def _write_body(tmp_path, text):
    path = tmp_path / "body.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_accepts_valid_body(tmp_path):
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, VALID_BODY))
    assert errors == []


def test_rejects_missing_heading(tmp_path):
    body = VALID_BODY.replace("## 其他信息\n无。\n", "")
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    assert any("missing or out-of-order heading" in error for error in errors)


def test_rejects_changed_checkbox_item(tmp_path):
    body = VALID_BODY.replace("- [ ] Bug 修复", "- [ ] Bug fix")
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    assert "类型 items differ from template" in errors


def test_rejects_template_prompt(tmp_path):
    body = VALID_BODY.replace("执行了测试。", "执行了测试。\n简要描述测试方案。")
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    assert any("template prompt remains" in error for error in errors)


def test_rejects_empty_precision_section(tmp_path):
    body = VALID_BODY.replace("| 项目 | 结果 |\n| --- | --- |\n| loss | 通过 |\n", "")
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    assert "section has no content: 精度和性能验证结果" in errors


def test_rejects_changed_heading_level(tmp_path):
    body = VALID_BODY.replace("## 描述", "### 描述")
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    assert any("missing or out-of-order heading" in error for error in errors)


def test_rejects_body_inside_code_fence(tmp_path):
    body = "```markdown\n" + VALID_BODY + "```\n"
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    assert any("missing or out-of-order heading" in error for error in errors)


def test_accepts_checkbox_only_section(tmp_path):
    body = VALID_BODY.replace("执行了测试。", "- [x] 运行 pytest\n- [x] 运行 lint")
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    assert errors == []


def test_accepts_content_under_subheading(tmp_path):
    body = VALID_BODY.replace("执行了测试。", "### 单元测试\n运行 pytest。")
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    assert errors == []


def test_ignores_template_prompt_inside_code_fence(tmp_path):
    body = VALID_BODY.replace("执行了测试。", "执行了测试。\n\n```\n简要描述测试方案。\n```")
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    assert errors == []


def test_template_prompts_are_recognized():
    template = TEMPLATE.read_text(encoding="utf-8")
    prompt_lines = [
        line.strip()
        for _, line in validate_pr_body.unfenced_lines(template)
        if line.strip() and not line.strip().startswith("#") and not validate_pr_body.CHECKBOX_RE.match(line)
    ]
    assert prompt_lines
    for line in prompt_lines:
        assert any(prompt in line for prompt in validate_pr_body.PROMPTS), line


def test_reports_out_of_order_heading_once(tmp_path):
    body = VALID_BODY.replace("## 描述\n说明改动。\n\n", "", 1)
    body = body.replace("## 如何测试", "## 描述\n说明改动。\n\n## 如何测试", 1)
    errors, _ = validate_pr_body.validate(TEMPLATE, _write_body(tmp_path, body))
    heading_errors = [error for error in errors if "heading" in error]
    assert heading_errors == ["out-of-order heading: ## 类型"]


def test_main_exit_codes(tmp_path, monkeypatch):
    valid = _write_body(tmp_path, VALID_BODY)
    invalid = tmp_path / "invalid.md"
    invalid.write_text("## 描述\n说明改动。\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["validate_pr_body.py", "--template", str(TEMPLATE), "--body", str(valid)])
    assert validate_pr_body.main() == 0

    monkeypatch.setattr(sys, "argv", ["validate_pr_body.py", "--template", str(TEMPLATE), "--body", str(invalid)])
    assert validate_pr_body.main() == 1
