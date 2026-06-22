# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the multiturn_chat patch (_patched_validate + _patched_tokenize)."""

from unittest.mock import MagicMock

import pytest
import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan_npu.patches.torchtitan.multiturn_chat import (
    _patched_tokenize,
    _patched_validate,
)


# ---------------------------------------------------------------------------
# _patched_validate
# ---------------------------------------------------------------------------


class TestPatchedValidate:
    def test_valid_single_turn(self):
        _patched_validate(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        )

    def test_valid_multi_turn(self):
        _patched_validate([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "ok"},
        ])

    def test_valid_with_tool_role(self):
        _patched_validate([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "tool_calls": []},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "got it"},
        ])

    def test_rejects_unknown_role(self):
        with pytest.raises(ValueError, match="Unknown role"):
            _patched_validate(
                [{"role": "user", "content": "hi"}, {"role": "bot", "content": "hello"}]
            )

    def test_rejects_single_message(self):
        with pytest.raises(ValueError, match="at least 2"):
            _patched_validate([{"role": "user", "content": "hi"}])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least 2"):
            _patched_validate([])

    # --- 2-message strict mode ---

    def test_2msg_rejects_system_first(self):
        with pytest.raises(ValueError, match="First message must be 'user'"):
            _patched_validate(
                [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "ok"}]
            )

    def test_2msg_rejects_user_user(self):
        with pytest.raises(ValueError, match="Second message must be 'assistant'"):
            _patched_validate(
                [{"role": "user", "content": "hi"}, {"role": "user", "content": "again"}]
            )

    # --- multi-turn relaxed mode ---

    def test_multiturn_allows_system_first(self):
        _patched_validate([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])

    def test_multiturn_rejects_assistant_first(self):
        with pytest.raises(ValueError, match="must be 'user' or 'system'"):
            _patched_validate([
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ])


# ---------------------------------------------------------------------------
# Hardcoded exact-match tests for Qwen3 multiturn chat with real tokenizer.
#
# In each test:
#  - input_ids = full_tokens[:-1]: the BOS-prefixed token sequence with the
#    last token stripped (<|im_end|> may be absent if it was the final token).
#  - label_ids = full_tokens[1:] with only assistant content tokens and their
#    closing <|im_end|> kept (≠ IGNORE_INDEX).  All format markers
#    (<|im_start|>, \n, role words) and non-assistant content are masked.
#
# Notice these important tokens:
# - 151644 = <|im_start|>
# - 151645 = <|im_end|> (= eos_id in Qwen3)
# - 151667 = <think>
# - 151668 = </think>
#
# <think> / </think> appear in the last assistant turn because Qwen3's chat
# template unconditionally wraps the final assistant message with
# <think>\n\n</think>\n\n even when reasoning_content is not provided.
# These format tokens are treated as part of the assistant span and kept
# in labels — the model must learn to emit them to follow the template.
# Non-last assistant turns use plain format (no think wrapper).
# -    271 = \n\n  (double newline, used in think wrapper)
# -    198 = \n    (single newline)
# - -100 = IGNORE_INDEX (masked label)
# ---------------------------------------------------------------------------


class TestHardcodedExactMatch:
    """Each test compares _patched_tokenize output against a hardcoded reference."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from pathlib import Path
        from transformers import AutoTokenizer

        tokenizer_path = Path("tests/assets/tokenizer/qwen3-tokenizer")
        if not (tokenizer_path / "tokenizer.json").exists():
            pytest.skip("Real tokenizer assets not available")
        self._tok = AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True
        )

    def _run(self, messages, seq_len=256):
        mock = MagicMock()
        mock.seq_len = seq_len
        mock._eos_id = self._tok.eos_token_id
        mock._sample_processor = MagicMock(return_value=messages)
        mock._tokenizer = self._tok
        mock._sample_idx = 0
        return _patched_tokenize(mock, {"sample": 1})

    def test_system_user_assistant(self):
        input_ids, label_ids = self._run([
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
        ])
        assert input_ids == [
            151644, 8948, 198, 3430, 10950, 13, 151645, 198,
            151644, 872, 198, 13048, 151645, 198,
            151644, 77091, 198, 151667, 271, 151668, 271, 9707, 13,
        ]
        assert label_ids == [
            -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100,
            -100, -100, 151667, 271, 151668, 271, 9707, 13, 151645,
        ]

    def test_two_assistant_turns(self):
        input_ids, label_ids = self._run([
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ])
        assert input_ids == [
            151644, 872, 198, 80, 16, 151645, 198,
            151644, 77091, 198, 64, 16, 151645, 198,
            151644, 872, 198, 80, 17, 151645, 198,
            151644, 77091, 198, 151667, 271, 151668, 271, 64, 17,
        ]
        assert label_ids == [
            -100, -100, -100, -100, -100, -100, -100,
            -100, -100,   64,   16, 151645, -100, -100,
            -100, -100, -100, -100, -100, -100, -100,
            -100, -100, 151667,  271, 151668,  271,   64,   17, 151645,
        ]

    def test_single_token_responses(self):
        input_ids, label_ids = self._run([
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "Y"},
            {"role": "user", "content": "z"},
            {"role": "assistant", "content": "W"},
        ])
        assert input_ids == [
            151644, 872, 198, 87, 151645, 198,
            151644, 77091, 198, 56, 151645, 198,
            151644, 872, 198, 89, 151645, 198,
            151644, 77091, 198, 151667, 271, 151668, 271, 54,
        ]
        assert label_ids == [
            -100, -100, -100, -100, -100, -100,
            -100, -100,   56, 151645, -100, -100,
            -100, -100, -100, -100, -100, -100, -100,
            -100, 151667,  271, 151668,  271,   54, 151645,
        ]

    def test_drop_sample_exceeds_seq_len(self):
        result = self._run(
            [{"role": "user", "content": "a" * 200},
             {"role": "assistant", "content": "b" * 200},
             {"role": "user", "content": "c" * 200}],
            seq_len=5,
        )
        assert result is None

    def test_first_label_always_masked(self):
        _, label_ids = self._run([
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "thanks"},
        ])
        assert label_ids[0] == IGNORE_INDEX
        kept = [lid for lid in label_ids if lid != IGNORE_INDEX]
        assert 151645 in kept, "<|im_end|> from assistant turn should be kept"

    def test_three_assistant_turns_long_dialogue(self):
        """7 messages, 3 assistant turns: [system, user, asst, user, asst, user, asst]."""
        input_ids, label_ids = self._run([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You are welcome!"},
        ])
        # system: You are a helpful assistant.
        # user1:  Hello
        # asst1:  Hi there!
        # user2:  What is Python?
        # asst2:  A programming language.
        # user3:  Thanks
        # asst3:  You are welcome!  (last → with <think> wrapper)
        assert input_ids == [
            151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
            151644, 872, 198, 9707, 151645, 198,
            151644, 77091, 198, 13048, 1052, 0, 151645, 198,
            151644, 872, 198, 3838, 374, 13027, 30, 151645, 198,
            151644, 77091, 198, 32, 15473, 4128, 13, 151645, 198,
            151644, 872, 198, 12658, 151645, 198,
            151644, 77091, 198, 151667, 271, 151668, 271, 2610, 525, 10565, 0,
        ]
        assert label_ids == [
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100,
            -100, -100, 13048, 1052, 0, 151645, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, 32, 15473, 4128, 13, 151645, -100,
            -100, -100, -100, -100, -100, -100,
            -100, -100, -100, 151667, 271, 151668, 271, 2610, 525, 10565, 0, 151645,
        ]


# ---------------------------------------------------------------------------
# 2-message delegation test (uses mock — upstream code calls tokenizer
# without tokenize=False, which crashes with the real Qwen3 tokenizer)
# ---------------------------------------------------------------------------


class TestTwoMessageDelegation:
    """2-message samples delegate to upstream _ORIG_TOKENIZE."""

    def test_delegates_and_returns_valid_tuple(self):
        mock = MagicMock()
        mock.seq_len = 256
        mock._eos_id = 151645
        mock._sample_processor = MagicMock(return_value=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        # Mock tokenizer: apply_chat_template returns a string (upstream expects this)
        mock._tokenizer = MagicMock()
        mock._tokenizer.apply_chat_template = MagicMock(
            return_value="<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>"
        )
        mock._tokenizer.encode = MagicMock(return_value=[100, 1, 2, 3, 151645])
        mock._validate_messages = MagicMock()
        mock._logged_first_sample = True
        mock._sample_idx = 0

        result = _patched_tokenize(mock, {"sample": 1})
        assert result is not None
        input_ids, label_ids = result
        assert len(input_ids) == len(label_ids)
        assert len(input_ids) > 0
