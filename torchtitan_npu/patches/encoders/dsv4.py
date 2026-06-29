# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from dataclasses import dataclass
from typing import Any, Literal

from torchtitan.tools.logging import logger

from .base import BaseChatEncoder, ChatEncoderConfig

VALID_THINKING_MODES = ("chat", "thinking")
VALID_REASONING_EFFORTS = ("max", "high")


class DSV4ChatEncoder(BaseChatEncoder):
    def __init__(
        self,
        encoding_module_path: str,
        thinking_mode: Literal["chat", "thinking"] = "thinking",
        drop_thinking: bool = True,
        reasoning_effort: Literal["max", "high"] | None = None,
    ):
        if thinking_mode not in VALID_THINKING_MODES:
            raise ValueError(f"thinking_mode must be one of {VALID_THINKING_MODES}, got {thinking_mode!r}")
        if reasoning_effort is not None and reasoning_effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {VALID_REASONING_EFFORTS} or None, got {reasoning_effort!r}"
            )
        spec = importlib.util.spec_from_file_location("encoding_dsv4", encoding_module_path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load encoding module from: {encoding_module_path}")
        self._enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._enc)
        self.thinking_mode = thinking_mode
        self.drop_thinking = drop_thinking
        self.reasoning_effort = reasoning_effort

        logger.info(
            "[DSV4ChatEncoder] Loaded encoding module from %s (thinking_mode=%s, drop_thinking=%s)",
            encoding_module_path,
            thinking_mode,
            drop_thinking,
        )

    @staticmethod
    def _adapt_messages(messages):
        adapted = []
        for msg in messages:
            msg = dict(msg)
            if "reasoning" in msg and "reasoning_content" not in msg:
                msg["reasoning_content"] = msg.pop("reasoning")
            adapted.append(msg)
        return adapted

    @staticmethod
    def _validate_assistant_range_alignment(
        *,
        idx: int,
        prefix_tokens: list[int],
        through_tokens: list[int],
        full_tokens: list[int],
    ) -> None:
        prefix_len = len(prefix_tokens)
        through_len = len(through_tokens)
        if full_tokens[:prefix_len] != prefix_tokens:
            raise ValueError(
                "DSV4 assistant span prefix alignment failed at message index "
                f"{idx}: prefix tokenization does not match full token prefix."
            )
        if full_tokens[:through_len] != through_tokens:
            raise ValueError(
                "DSV4 assistant span boundary alignment failed at message index "
                f"{idx}: through tokenization does not match full token prefix."
            )

    def encode_messages_with_assistant_ranges(self, messages: list[dict[str, Any]], tokenizer, eos_id):
        adapted = self._adapt_messages(messages)
        prepared = self._prepare_messages(adapted)
        rendered_messages = self._render_messages(prepared)
        full_text = "".join(rendered_messages).rstrip("\n")
        full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != eos_id:
            full_tokens.append(eos_id)

        assistant_ranges = []
        prefix_text = ""
        for idx, (msg, rendered) in enumerate(zip(prepared, rendered_messages, strict=True)):
            next_prefix_text = prefix_text + rendered
            prefix_tokens = tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
            through_tokens = tokenizer.encode(next_prefix_text, add_bos=True, add_eos=False)
            if msg.get("role") == "assistant":
                start = len(prefix_tokens)
                end = len(through_tokens)
                self._validate_assistant_range_alignment(
                    idx=idx,
                    prefix_tokens=prefix_tokens,
                    through_tokens=through_tokens,
                    full_tokens=full_tokens,
                )
                assistant_ranges.append((start, end))
            prefix_text = next_prefix_text

        return full_text, full_tokens, assistant_ranges

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = self._enc.merge_tool_messages(messages)
        prepared = self._enc.sort_tool_results_by_call_order(prepared)

        effective_drop_thinking = self.drop_thinking
        if any(msg.get("tools") for msg in prepared):
            effective_drop_thinking = False

        if self.thinking_mode == "thinking" and effective_drop_thinking:
            prepared = self._enc._drop_thinking_messages(prepared)
        return prepared

    def _render_messages(self, messages: list[dict[str, Any]]) -> list[str]:
        effective_drop_thinking = self.drop_thinking
        if any(msg.get("tools") for msg in messages):
            effective_drop_thinking = False

        return [
            self._enc.render_message(
                idx,
                messages,
                thinking_mode=self.thinking_mode,
                drop_thinking=effective_drop_thinking,
                reasoning_effort=self.reasoning_effort if idx == 0 else None,
            )
            for idx in range(len(messages))
        ]


@dataclass(kw_only=True, slots=True)
class DSV4EncoderConfig(ChatEncoderConfig):
    encoding_module_path: str = ""
    thinking_mode: Literal["chat", "thinking"] = "thinking"
    drop_thinking: bool = True
    reasoning_effort: Literal["max", "high"] | None = None

    def __post_init__(self):
        if self.thinking_mode not in VALID_THINKING_MODES:
            raise ValueError(f"thinking_mode must be one of {VALID_THINKING_MODES}, got {self.thinking_mode!r}")
        if self.reasoning_effort is not None and self.reasoning_effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {VALID_REASONING_EFFORTS} or None, got {self.reasoning_effort!r}"
            )

    def build(self) -> DSV4ChatEncoder:
        return DSV4ChatEncoder(
            encoding_module_path=self.encoding_module_path,
            thinking_mode=self.thinking_mode,
            drop_thinking=self.drop_thinking,
            reasoning_effort=self.reasoning_effort,
        )
