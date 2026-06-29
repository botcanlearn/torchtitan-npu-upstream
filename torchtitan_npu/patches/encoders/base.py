# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BaseChatEncoder and ChatEncoderConfig base classes."""

from dataclasses import dataclass
from typing import Any

AssistantRanges = list[tuple[int, int]]


class BaseChatEncoder:
    """Base class for custom chat encoders.

    Use this when tokenizer.apply_chat_template() is not sufficient for a
    model or dataset, for example when rendering needs model-specific tool
    message merging, thinking/chat mode switching, or non-ChatML markers.

    ChatDataset requires custom encoders to return the rendered text, the
    tokenized sequence, and the assistant token ranges used for label masking.
    """

    def encode_messages_with_assistant_ranges(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Any,
        eos_id: int,
    ) -> tuple[str, list[int], AssistantRanges]:
        raise NotImplementedError(f"{type(self).__name__}.encode_messages_with_assistant_ranges() not implemented")


@dataclass(kw_only=True, slots=True)
class ChatEncoderConfig:
    encoding_module_path: str = ""

    def build(self) -> BaseChatEncoder:
        raise NotImplementedError(f"{type(self).__name__}.build() not implemented")
