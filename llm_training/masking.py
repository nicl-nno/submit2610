'Masking: prompt/reference boundaries and tool-token masking.\n\n\nTwo concerns live here\n\n1. **Prompt vs reference boundary** (all task classes). The reference answer is\n   *not* an SFT label in GRPO, but its token span is needed: for the\n   length-discard rules (32 000 total / 8 000 reference tokens), for the\n   LLM-as-Judge input, and as an SFT-fallback. The boundary is found by\n   *differential rendering*: render ``prompt`` with ``add_generation_prompt=True``\n   and the full conversation with ``add_generation_prompt=False``, then take\n   the token suffix beyond the prompt prefix.\n\n2. **Tool-token masking** (tool-use classes only). For ``opencode_gen`` and\n   ``tool_calls_multiturn``, AGENTS.md requires masking ``<tool_call>`` /\n   ``<tool_response>`` tokens so the reward (and the future GRPO loss) shapes\n   only the final answer, not the tool mechanics. The mask produced here is the\n   data the future GRPO-loop masking policy will consume.\n   ``<tool_call>``/``<tool_response>``'

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_training.schema import UnifiedExample
from llm_training.tokenizer_utils import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    prepare_messages,
    render,
    tokenize,
)

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer

# Matches a full <tool_call>…</tool_call> or <tool_response>…</tool_response>
# block (DOTALL so the contents may span newlines).

_TOOL_BLOCK_RE = re.compile(
    rf"{TOOL_CALL_OPEN}.*?{TOOL_CALL_CLOSE}|{TOOL_RESPONSE_OPEN}.*?{TOOL_RESPONSE_CLOSE}",
    re.DOTALL,
)


@dataclass
class MaskResult:
    'Outcome of masking a single :class:`UnifiedExample`.\n\n\nAttributes\n        prompt_ids: tokens the policy conditions on (TRL ``prompt`` rendered).\n        reference_ids: tokens of the reference answer (the label suffix).\n        total_tokens: ``len(prompt_ids) + len(reference_ids)``.\n        prefix_ok: whether the full token sequence starts with the prompt\n            tokens (i.e. differential split is clean). When ``False`` the\n            example is malformed and should be discarded.\n        labels: SFT-style labels (``-100`` on prompt, ids on reference).\n        tool_token_mask: bool list over ``reference_ids``, ``True`` for tokens\n            inside ``<tool_call>``/``<tool_response>`` blocks; ``None`` for\n            non-tool examples.'

    prompt_ids: list[int]
    reference_ids: list[int]
    total_tokens: int
    prefix_ok: bool
    labels: list[int]
    tool_token_mask: list[bool] | None


def mask_example(
    example: UnifiedExample,
    tokenizer: "PreTrainedTokenizer",
) -> MaskResult:
    'Compute prompt/reference token spans and the tool-token mask.'
    prompt_messages, full_messages = prepare_messages(example, tokenizer)

    prompt_text = render(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = render(tokenizer, full_messages, add_generation_prompt=False)

    prompt_ids = tokenize(tokenizer, prompt_text)
    full_ids = tokenize(tokenizer, full_text)

    n_prompt = len(prompt_ids)
    prefix_ok = full_ids[:n_prompt] == prompt_ids
    if prefix_ok:
        reference_ids = full_ids[n_prompt:]
    else:
        # Fallback: re-tokenize the textual suffix. This still gives a usable
        # reference-id list even if the clean prefix split broke at a token
        # boundary, but the example is flagged (prefix_ok=False) so callers can
        # discard it.
        
        reference_text = full_text[len(prompt_text) :] if full_text.startswith(prompt_text) else ""
        reference_ids = tokenize(tokenizer, reference_text)

    labels = [-100] * n_prompt + list(reference_ids)

    is_tool_use = example.tools is not None and len(example.tools) > 0
    if is_tool_use:
        # Reuse the already-rendered prompt_text / full_text instead of
        # re-rendering them (4 renders → 2). The textual suffix is the rendered
        # reference span (with <tool_call> markers), which is exactly what
        # compute_tool_token_mask needs to localise tool-call blocks.
        
        
        reference_text = full_text[len(prompt_text) :] if full_text.startswith(prompt_text) else ""
        tool_token_mask = compute_tool_token_mask(reference_text, tokenizer)
        if len(tool_token_mask) != len(reference_ids):
            # Align lengths defensively (should not happen for well-formed
            # examples; trim/pad to len(reference_ids)).
            tool_token_mask = tool_token_mask[: len(reference_ids)] + [False] * (
                len(reference_ids) - len(tool_token_mask)
            )
    else:
        tool_token_mask = None

    return MaskResult(
        prompt_ids=prompt_ids,
        reference_ids=reference_ids,
        total_tokens=n_prompt + len(reference_ids),
        prefix_ok=prefix_ok,
        labels=labels,
        tool_token_mask=tool_token_mask,
    )


def compute_tool_token_mask(
    reference_text: str,
    tokenizer: "PreTrainedTokenizer",
) -> list[bool]:
    'Mark tokens of ``reference_text`` that fall inside a tool block.\n\n\n    Returns a bool list of length ``len(tokenize(reference_text))`` where\n    ``True`` means the token lies within a ``<tool_call>…</tool_call>`` or\n    ``<tool_response>…</tool_response>`` span (delimiters included).'
    enc = tokenizer(reference_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets: list[tuple[int, int]] = enc["offset_mapping"]
    # Char spans of every tool block
    spans: list[tuple[int, int]] = [(m.start(), m.end()) for m in _TOOL_BLOCK_RE.finditer(reference_text)]

    def _overlaps(tok_start: int, tok_end: int) -> bool:
        for s, e in spans:
            if tok_start < e and s < tok_end:
                return True
        return False

    return [_overlaps(s, e) for (s, e) in offsets]
