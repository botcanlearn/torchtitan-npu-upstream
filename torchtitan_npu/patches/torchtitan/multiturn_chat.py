# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/hf_datasets/text_datasets.py#L302-L315
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Monkey-patch ChatDataset for multi-turn conversation support.

Upstream torchtitan's ChatDataset validates exactly 2 messages ([user, assistant]).
Wordle dataset have more than 2 messages ([system, user, assistant, user, assistant, ...]).

This module monkey-patches ChatDataset._tokenize_sample to support multi-turn
messages. It also provides _patched_validate, a relaxed message validator called
directly by _patched_tokenize for messages with >2 turns.

"""

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets import text_datasets
from torchtitan.tools.logging import logger as _logger

# pyrefly: ignore [protected-access]
_ORIG_TOKENIZE = text_datasets.ChatDataset._tokenize_sample


# pyrefly: ignore [invalid-decorator]
@staticmethod
def _patched_validate(messages: list[dict]) -> None:
    """Relax single-turn restriction to allow multi-turn conversations.

    Keeps upstream's role-ordering rules but drops the ``len(messages) == 2``
    hard-cap so that multi-turn [user, assistant, user, assistant, ...] is
    accepted.
    """
    for i, msg in enumerate(messages):
        if msg["role"] not in ("user", "assistant", "system", "tool"):
            raise ValueError(f"Unknown role '{msg['role']}' at index {i}")

    if len(messages) < 2:
        raise ValueError(f"Expected at least 2 messages, got {len(messages)}")

    # For the 2-message case, keep upstream's strict checks so existing
    # single-turn tests continue to pass.
    if len(messages) == 2:
        if messages[0]["role"] != "user":
            raise ValueError(f"First message must be 'user', got '{messages[0]['role']}'")
        if messages[1]["role"] != "assistant":
            raise ValueError(f"Second message must be 'assistant', got '{messages[1]['role']}'")
    else:
        # Multi-turn: allow [system, user, assistant, ...]
        if messages[0]["role"] not in ("user", "system"):
            raise ValueError(f"First message must be 'user' or 'system', got '{messages[0]['role']}'")


def _patched_tokenize(self, sample: dict) -> tuple[list[int], list[int]] | None:
    """Multi-turn-aware tokenization: mask non-assistant spans in label_ids."""
    messages = self._sample_processor(sample)  # pyrefly: ignore [protected-access]

    if len(messages) == 2:
        # Original _tokenize_sample calls _validate_messages internally;
        # it now points to _patched_validate which enforces role rules
        # while still rejecting 2-message format errors.
        return _ORIG_TOKENIZE(self, sample)

    # Multi-turn path — _ORIG_TOKENIZE would reject >2 messages, so we
    # handle validation and tokenization ourselves.
    _patched_validate(messages)

    # Two-step tokenization: render the template to a string, then encode
    # it ourselves.  We can't use apply_chat_template(tokenize=True) because
    # Qwen3's tokenizer config has add_bos_token=False — internal tokenization
    # would skip BOS, and we wouldn't see the raw token list to scan for
    # assistant spans below.
    #
    # Also, Qwen3's template appends "\n" after every <|im_end|>, including the
    # last one, so we rstrip it.
    full_text = self._tokenizer.apply_chat_template(messages, tokenize=False).rstrip(
        "\n"
    )  # pyrefly: ignore [protected-access]

    # The conditional EOS-append below is a no-op for Qwen3, but a safety
    # net for tokenizers where the template terminator ≠ eos).
    full_tokens = self._tokenizer.encode(  # pyrefly: ignore [protected-access]
        full_text, add_bos=True, add_eos=False
    )
    if full_tokens[-1] != self._eos_id:  # pyrefly: ignore [protected-access]
        full_tokens.append(self._eos_id)  # pyrefly: ignore [protected-access]

    if len(full_tokens) - 1 > self.seq_len:
        _logger.debug(f"Dropping sample {self._sample_idx}: tokens exceeds seq_len {self.seq_len}")
        return None

    input_ids = full_tokens[:-1]
    label_ids = full_tokens[1:]
    eos_id = self._eos_id  # pyrefly: ignore [protected-access]

    # Build is_assistant on full_tokens (not input_ids) so that <|im_end|>
    # (which may be the last token, stripped from input_ids) is included.
    # label_ids[i] = full_tokens[i+1], so we mask by checking
    # is_assistant_full[i+1].
    is_assistant_full = [0] * len(full_tokens)

    im_end_id = eos_id  # Qwen3's <|im_end|> is eos
    assistant_header = self._tokenizer.encode(
        "<|im_start|>assistant\n", add_bos=False, add_eos=False
    )  # pyrefly: ignore [protected-access]
    assistant_header_len = len(assistant_header)
    pos = 0
    while pos < len(full_tokens):
        if (
            pos + assistant_header_len <= len(full_tokens)
            and full_tokens[pos : pos + assistant_header_len] == assistant_header
        ):
            resp_start = pos + assistant_header_len
            resp_end = resp_start
            while resp_end < len(full_tokens) and full_tokens[resp_end] != im_end_id:
                resp_end += 1
            # Mark assistant content + closing <|im_end|> (inclusive).
            # The model must learn to emit <|im_end|> to stop generation.
            for p in range(resp_start, min(resp_end + 1, len(full_tokens))):
                is_assistant_full[p] = 1
            pos = resp_end + 1
        else:
            pos += 1

    # label_ids[i] = full_tokens[i+1]; keep if that token is assistant content.
    for i, _ in enumerate(label_ids):
        if not is_assistant_full[i + 1]:
            label_ids[i] = IGNORE_INDEX

    return input_ids, label_ids


# Apply patches
text_datasets.ChatDataset._tokenize_sample = _patched_tokenize  # pyrefly: ignore [protected-access]
_logger.info("[Patch] Multi-turn ChatDataset enabled")
