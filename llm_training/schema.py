"Unified dataset schema — TRL-compatible example with task-class metadata.\n\n\nThe unified dataset is consumed by TRL's ``GRPOTrainer``. A row must carry a\nconversational ``prompt`` column (``list[{role, content}]``); every extra column\nis preserved by ``GRPOConfig`` (``remove_unused_columns=False`` by default) and\nforwarded as a keyword argument to the reward function — so ``task_class`` and\n``reference`` reach the multi-task reward dispatcher automatically.\n\n``task_class``"

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A chat message in the Qwen2.5 / Meno-Lite chat format.
# Meno-Lite.
Message = dict[str, Any]


# Canonical list of all 29 task classes. The reward function branches on this
# string and ``grpo_train.py``'s ``--val-task`` / ``--test-task`` choose hold-outs
# among them.


TASK_CLASSES: tuple[str, ...] = (
    # --- ShareGPT catalog (9)
    "coder",
    "creative_writer",
    "data_analyst",
    "deep_research",
    "factual_qa",
    "instruction_following",
    "math_reasoner",
    "rewriter",
    "summarizer",
    # --- Menon QA (1) ---
    "menon_qa",
    # --- mt-RAGEval (1, 8 sub-types merged)
    "mtrageval",
    # --- RAGU ICL (6) ---
    "ragu_artifact_extraction",
    "ragu_artifact_validation",
    "ragu_entity_extraction",
    "ragu_entity_validation",
    "ragu_relation_extraction",
    "ragu_relation_validation",
    # --- Other sources (one class each)
    "asr_summarization",
    "tool_calls_multiturn",
    "multihop_rag",
    "rummlu_pro",
    "self_cognition",
    "opencode_gen",
    # --- Meno-stand distillation (3; GPT-4o through the AU "Menon") ---
    
    "meno_rerank",
    "meno_rewrite",
    "meno_final",
    # --- DocSem quantitative reasoning (1; distilled check-181) ---
    
    "docsem_quant",
    # --- NEREL_Instruct (HF runtime; 6 configs merged)
    "nerel_instruct",
    # --- Data B: distilled BenchmarkAgent loop (spec 0.2.1, decision №21) ---
    
    "distill_b_loop",
)
# NOTE (frozen contract): the GRPO-EXTRACTION line's classes (3 extraction
# stages, 3 summary kinds, fineweb_qa) deliberately do NOT join this
# registry: TASK_CLASSES gates UnifiedExample (which requires a reference —
# the extraction corpus is reference-free by design), the loaders, and
# run_eval.py's default task list (no loaders/scoring for them there).
# The extraction line dispatches by its own constants
# (llm_training.extraction.prompts.STAGES, .summary_prompts.SUMMARY_STAGES,
# .qa_prompts.QA_TASK_CLASS); RewardConfig's fluency allowlists validate
# against TASK_CLASSES ∪ EXTRACTION_LINE_TASK_CLASSES (see below).
assert len(TASK_CLASSES) == 29, f"Expected 29 task classes, got {len(TASK_CLASSES)}"

#: Extra task classes of the GRPO-EXTRACTION line (the 8-class design): the 3
#: consolidation kinds + the QA class. The 3 extraction stage classes reuse
#: existing TASK_CLASSES names (ragu_*_extraction), so they are not listed
#: here. These classes DO NOT join TASK_CLASSES (see the frozen-contract note
#: above), but RewardConfig's fluency allowlists accept them on top of
#: TASK_CLASSES (validation = TASK_CLASSES ∪ this tuple) — consolidation and
#: QA are NL tasks where the grammar-classifier signal applies.


EXTRACTION_LINE_TASK_CLASSES: tuple[str, ...] = (
    "ragu_entity_summary",
    "ragu_relation_summary",
    "ragu_community_summary",
    "fineweb_qa",
)
assert len(set(EXTRACTION_LINE_TASK_CLASSES)) == 4


@dataclass
class UnifiedExample:
    "One row of the unified dataset, ready to feed TRL ``GRPOTrainer``.\n\n\nAttributes\n        task_class: task-class identifier (one of :data:`TASK_CLASSES`).\n            Identifies the class of the task, for reward dispatch & hold-out split.\n        prompt: conversational prompt = chat messages WITHOUT the reference\n            assistant turn. TRL applies the Meno-Lite chat template to it.\n        reference: the human reference answer (last assistant content for plain\n            tasks; the full assistant-generated content for tool-use tasks).\n        tools: tool schemas for the tool-use classes (``opencode_gen``,\n            ``tool_calls_multiturn``), ``None`` otherwise. Kept as structured\n            metadata for the future GRPO-loop tool executor; the tools are\n            baked into ``prompt``'s system message for TRL rendering.\n        reference_messages: assistant/tool turns appended to ``prompt`` to\n            reconstruct the full conversation (used for masking & rendering).\n        source: provenance string (file path / HF id / sub-type).\n        extra: optional per-source metadata (e.g. mtrageval sub-type, ragu\n            quality_rating, nerel config)."

    task_class: str
    prompt: list[Message]
    reference: str
    tools: list[Message] | None = None
    reference_messages: list[Message] = field(default_factory=list)
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_class not in TASK_CLASSES:
            raise ValueError(
                f'Unknown task_class {self.task_class!r}; must be one of {TASK_CLASSES}'
            )
        if not self.reference:
            raise ValueError(
                'UnifiedExample.reference must be non-empty.\nUnifiedExample.reference'
            )
        if not isinstance(self.prompt, list) or not self.prompt:
            raise ValueError(
                'UnifiedExample.prompt must be a non-empty list of messages.\nUnifiedExample.prompt'
            )


def to_dataset_row(example: UnifiedExample) -> dict[str, Any]:
    'Flatten a :class:`UnifiedExample` into a ``datasets.Dataset`` row.\n\n\n    Extra columns (``task_class``, ``reference``, ``tools``, ``source``,\n    ``extra``) are preserved by ``GRPOConfig`` and reach the reward function.'
    return {
        "prompt": list(example.prompt),
        "task_class": example.task_class,
        "reference": example.reference,
        "tools": example.tools,
        "reference_messages": list(example.reference_messages),
        "source": example.source,
        "extra": dict(example.extra),
    }
