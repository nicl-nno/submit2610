'Configuration + the scorer protocol for the reward layer.\n\n\nTwo abstractions live here\n\n* :class:`RewardConfig` — the reusable reward configuration (in-process\n  embeddings + optional judge HTTP endpoint, judge-resilience knobs, known tool\n  names). Shared by ``run_eval.py`` and the future ``grpo_train.py``.\n:class:`RewardConfig`\n* :class:`Scorer` / :class:`ScoredMetrics` — the interface the dispatcher depends\n  on. The real RAGAS-backed implementation lives in :mod:`llm_training.reward.judge`;\n  tests inject a fake. This keeps RAGAS out of the import graph of the pure\n  dispatch logic.\n\nReward philosophy (revised)\n\nThe reward combines per-example signals, unified across all 25 task\nclasses (no more RAG vs non-RAG split):\n\n* **Semantic Similarity** — always (RAGAS ``AnswerSimilarity``; embeddings only,\n  no LLM). Cheap, deterministic, the workhorse in-loop reward during GRPO.\n* **Answer Correctness** — only when a judge endpoint is configured (RAGAS\n  ``AnswerCorrectness``; LLM). The semantic anchor for ctrl-points / final eval.\n* **Format check** — only for tool-use classes (validates ``<tool_call>``\n  structure; pure Python, no network).\n\nActive signals are combined via the harmonic mean (a zero on any active\ndimension, e.g. an invalid tool-call, zeroes the reward). ``Faithfulness`` and\n``NoiseSensitivity`` were removed: they re-sent the full retrieved context in\neach of 4–5 internal RAGAS calls, which made per-example judge traffic ~5–10×\nlarger and added intolerable latency to the GRPO loop.'

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llm_training.runtime.endpoints import OpenAIEndpoint
from llm_training.schema import EXTRACTION_LINE_TASK_CLASSES, TASK_CLASSES


#: Default in-process embedding model (bge-m3, multilingual, 1024-dim).

DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-m3"

#: Default in-process fluency classifier: a grammar classifier (binary
#: "grammatical" / "ungrammatical") built on snowflake-arctic-embed-xs (a tiny
#: BERT: hidden 384, 6 layers, ~22M params). Trained on English C4 grammar data,
#: but transfers to Russian well (see the fluency demo notebooks) and produces
#: well-spread, reality-matching scores — unlike RuCoLa, whose P(acceptable) was
#: overconfident and squashed. The positive ("grammatical") label sits at index 1.

# : «grammatical»/«ungrammatical»




DEFAULT_LOCAL_FLUENCY_MODEL = "agentlans/snowflake-arctic-xs-grammar-classifier"

#: Default task classes where the fluency signal applies (natural-language
#: answers + the ``coder`` prose explanation). Tool-call turns, ``coder`` code
#: blocks (validated by ``code_check``), and short-answer / code-only classes
#: are excluded. ``coder`` is included because its answers mix prose + code and
#: the prose part benefits from a fluency signal. ``factual_qa`` is excluded:
#: its answers are terse one-liners that carry no prose-quality signal once the
#: per-sentence word-count gate lets single sentences through. ``nerel_instruct``
#: is included but gated to its prose sub-configs via
#: :data:`DEFAULT_FLUENCY_SUBTASKS` (definitions / normalization).

DEFAULT_FLUENCY_CLASSES: frozenset[str] = frozenset(
    {
        "menon_qa", "multihop_rag", "mtrageval",
        "creative_writer", "rewriter", "summarizer",
        "asr_summarization", "instruction_following", "self_cognition",
        "deep_research", "coder", "nerel_instruct",
        # The Meno-stand final answers are RU prose (a close kin of menon_qa);
        # meno_rewrite stays OUT (1-2 short search queries — the word-count
        # gate would drop them anyway), meno_rerank is a JSON verdict.
        
        # JSON.
        "meno_final",
    }
)

#: Default sub-config allowlist for classes that carry heterogeneous sub-tasks
#: under one ``task_class`` (keyed by ``extra["config"]``). ``nerel_instruct``
#: has 6 configs: the prose-bearing ``entity_definition`` /
#: ``relation_definition`` (always) and ``entity_normalization`` (conditional —
#: short normalizations are naturally dropped by the word-count gate) get
#: fluency; the deterministic extraction configs (``entity_recognition``,
#: ``relation_extraction``, ``relation_keywords``) are scored by a deterministic
#: F1 instead. A class absent from this map is governed solely by
#: :data:`DEFAULT_FLUENCY_CLASSES`.

DEFAULT_FLUENCY_SUBTASKS: dict[str, frozenset[str]] = {
    "nerel_instruct": frozenset(
        {
            "entity_definition",
            "relation_definition",
            "entity_normalization",
        }
    ),
}


@dataclass(frozen=True)
class RewardConfig:
    'Reusable reward configuration (in-process embeddings + optional judge).\n\n\n    Embeddings are computed **in-process** via ``sentence-transformers``\n    (no HTTP server, no vLLM, no cloud endpoint). The model is loaded by RAGAS\'s\n    ``HuggingFaceEmbeddings`` on the chosen device. The judge LLM — a different\n    model from the policy — stays an external OpenAI-compatible HTTP endpoint.\n\nArgs\n        local_embedding_model: HuggingFace model id for in-process embedding\n            inference via ``sentence-transformers``. Defaults to\n            ``BAAI/bge-m3`` (multilingual, 1024-dim). The model is downloaded\n            from HuggingFace on first use. HuggingFace-id\n        local_embedding_device: device for the local embedding model\n            (``"cuda"``, ``"cuda:0"``, ``"cpu"``, or ``None`` for auto-detect).\n            ``None`` lets ``SentenceTransformer`` auto-detect CUDA. On a GPU\n            with spare memory, ``"cuda"`` gives ~5–20 ms per embedding (vs\n            0.5–2 s on CPU\n        judge: OpenAI-compatible endpoint of the **judge LLM** (≠ policy), or\n            ``None`` to disable Answer Correctness (the in-loop / online reward\n            path during GRPO). ``None`` ⇒\n        compute_answer_correctness: when ``True`` (and ``judge`` is set), compute\n            Answer Correctness as part of ``combined``. Set to ``False`` to score\n            with embeddings alone even when a judge is configured (useful for\n            ablations). Ignored when ``judge is None``.\n        known_tool_names: registry of valid tool names for :func:`format_check`,\n            collected from the unified dataset at build time. Empty set disables\n            the name-membership sub-check (the JSON + fields sub-checks still\n        known_tool_schemas: registry ``{tool name -> parameter JSON schema}``\n            for :func:`llm_training.reward.schema_check.schema_check`, collected\n            from the unified dataset at build time by\n            ``collect_tool_schemas`` (the schema-level sibling of\n            ``known_tool_names``). Empty dict disables the check entirely (it\nreturns ``None``\n        enable_schema_check: master switch for the deterministic tool-call\n            ``schema_check`` (argument validation against\n            :attr:`known_tool_schemas` — required fields 0.5 + declared types\n            0.5, a stdlib-only JSON-Schema subset). Active only for tool-CALL\n            turns of tool-use classes. ``False`` (default) keeps the reward\n            exactly as before — the GRPO **anchors** never set it (the frozen\n            anchor methodology) and ``run_eval.py`` opts in via\n             ``--schema-check``; ``grpo_train.py`` enables it in the IN-LOOP\n             reward only (escape ``--no-schema-check``\n        enable_language_guard: master switch for the fluency **language guard**\n            (:func:`llm_training.reward.language.language_match_factor`): when\n            ``True``, the fluency score of a natural-language answer is\n            multiplied by the confidence of the example\'s **target language**\n            (``extra["target_language"]``, computed at build time) over the\n            generated prose — a reward-hacking defence against the policy\n            drifting to English (the grammar classifier is English-trained and\n            may score English slightly higher). ``False`` (default) keeps the\n            reward as before; ``grpo_train.py`` enables it in the IN-LOOP reward\n            only (escape ``--no-language-guard``) and ``run_eval.py`` opts in\n            via ``--language-guard`` (the frozen anchors never set it).\n        enable_rummlu_em: master switch for the deterministic ruMMLU-pro letter\n            exact match (:func:`llm_training.reward.rummlu_em.rummlu_em`).\n            When ``True`` (default), the ``rummlu_pro`` class is scored by the\n            option-letter comparison INSTEAD of Semantic Similarity (SemSim\n            over a single letter barely separates the correct option — the\n            NEREL-F1 / docsem replacement pattern). Safe for the frozen anchor\n            methodology: ``rummlu_pro`` is never a val/test hold-out class.\n        enable_rerank_em: master switch for the deterministic Meno-stand\n            rerank grading (:func:`llm_training.reward.rerank_em.rerank_em`):\n            0.3 for a parseable JSON verdict ``{"label": N}`` whose label —\n            integer or string — normalizes to N ∈ {0,1,2} + 0.7 scaled by\n            the ORDINAL label distance (exact → +0.7, an adjacent grade\n            |Δ|=1 → +0.35, the inverted verdict 0↔2 → +0). When\n            ``True`` (default), the ``meno_rerank`` class REPLACES Semantic\n            Similarity with it; Answer Correctness is structurally skipped\n            for the class (a judge over a 15-character JSON verdict would be\n            pure noise). The class carries fixed val rows (the stand\'s test\n            split) — the metric doubles as the anchor val\'s structural\n            signal for those rows\n        format_epsilon_floor: when > 0, the ``format_check`` component is\n            floored to this value BEFORE the strict harmonic combine —\n            ``max(format_score, floor)`` — so a broken tool-call envelope no\n            longer zeroes the WHOLE example (the r2 zero-variance lesson:\n            ``frac_reward_zero_std`` failed at 0.375–0.675). The floor keeps a\n            proportional penalty inside format (0 → ε, the harmonic mean still\n            drags the combined down) while letting the other components carry\n            their gradient. ``0.0`` (default) = off — the strict semantics\n            everywhere; the GRPO **in-loop** config alone sets it\n            (``grpo_train.py --format-epsilon``, recommendation 0.1–0.15),\n            the anchor / run_eval configs never do (the frozen measure stays\n        judge_concurrency: max in-flight judge requests when scoring a batch.\n        embedding_concurrency: max concurrent **in-process embedding**\n            computations (SemSim runs on every example; in GRPO it runs on\n            every completion of every step). The embeddings share the GPU with\n            the policy, so this caps GPU contention. When ``None`` (default),\n            falls back to ``judge_concurrency``. Enforced by an independent\n            semaphore inside :class:`~llm_training.reward.judge.RAGASScorer`,\n            so the judge endpoint is never throttled by the embedding limit (or\n            vice versa\n            ``None`` → ``judge_concurrency``.\n        judge_timeout_s: per-request timeout (seconds) for judge HTTP calls.\n        judge_max_tokens: max generation tokens for the judge LLM\'s structured\n            output. RAGAS defaults to 1024 which truncates the verbose\n            ``CorrectnessClassifier`` JSON; 4096 is RAGAS\'s own recommended\n        judge_max_retries: extra attempts (with exponential backoff) when a judge\n            call fails transiently (HTTP 429/5xx, proxy 400, network). 0\n            disables retries on top of the openai client\'s own retries\n        local_fluency_model: HuggingFace model id for the in-process fluency\n            classifier (default ``agentlans/snowflake-arctic-xs-grammar-classifier``,\n            a binary grammar classifier built on a tiny BERT encoder). Loaded via\n            ``transformers``; shares the policy GPU. Only used when\n            ``enable_fluency`` is True. HuggingFace-id\n        local_fluency_device: device for the fluency model (``"cuda"``, ``"cpu"``,\n            or ``None`` for auto-detect\n        fluency_concurrency: max concurrent **in-process** fluency forward\n            passes (per-sentence classification). When ``None`` (default), falls\n            back to ``embedding_concurrency`` (the fluency model shares the GPU\n            with the embedder and the policy). Enforced by an independent\n            semaphore inside the fluency scorer\n        fluency_classes: task classes where the fluency signal is active\n            (default :data:`DEFAULT_FLUENCY_CLASSES`). Classes outside this set\n            never compute fluency\n        fluency_calibrator_path: path to the JSON artifact produced once by\n            ``scripts/calibrate_fluency.py`` (an isotonic-regression calibrator\n            fit on GLM-5.2 fluency labels). ``None`` → no calibration (raw\n            P(grammatical) used directly) — the default, since the grammar\n            classifier\'s raw scores are already well-spread.\n        enable_fluency: master switch for the fluency signal. ``False`` disables\n            it entirely (no model load\n        enable_code_check: master switch for the ``code_check`` signal (Python\n            ``ast`` validation of ``coder`` answers). ``False`` disables it.\n        enable_nerel_f1: master switch for the deterministic NEREL F1 signal\n            (set-level F1 over normalized entity/relation lines, ported from\n            the EleutherAI NEREL-bench). When ``True``, the NEREL extraction\n            sub-configs (``entity_recognition``, ``relation_extraction``,\n            ``relation_keywords``, dispatched via ``extra["config"]``) are\n            scored by F1 INSTEAD of Semantic Similarity — SemSim is skipped\n            entirely for them (no embedding compute). ``False`` leaves them on\n            the default SemSim path\n        enable_docsem_check: master switch for the deterministic DocSem check\n            (:func:`llm_training.reward.docsem.docsem_check` — executable\n            Program-of-Thought scoring of the ``docsem_quant`` class against\n            the gold answer/evidence from ``extra``: the JSON format, the\n            program executed in the safe AST-whitelist interpreter, the\n            canonical answer equality and the evidence set-F1, combined by\n            the strict harmonic mean). When ``True`` (default), the class is\n            scored by it INSTEAD of Semantic Similarity — SemSim is skipped\n            entirely (no embedding compute), exactly like the NEREL F1\n            pattern. ``False`` leaves the class on the default SemSim path.\n        enable_reasoning_semsim: when ``False`` (default), SGR-agent **reasoning\n            turns** (``extra["is_reasoning_turn"]``, the open-ended CoT phase of\n            the sgr-agent-core framework) are scored by fluency + the SGR-schema\n            structural check (:func:`~llm_training.reward.scoring.reasoning_check`)\n            — NOT by Semantic Similarity. Rationale: reasoning is open-ended\n            (infinitely many valid reasonings reach the same conclusion), so\n            SemSim would reward paraphrase-imitation of the teacher\'s wording\n            rather than the Schema-Guided-Reasoning structure; the structural\n            check rewards emitting that structure (``Reasoning steps`` /\n            ``Current situation`` / ``Plan status`` / ``Remaining steps``). Set\n            ``True`` to score reasoning turns with SemSim like other turns (an\n            escape hatch for ablation\n        enable_judge_rubrics: when ``True`` AND a judge endpoint is configured,\n            SGR-agent **reasoning turns** (where Answer Correctness is\n            structurally skipped — reasoning is open-ended) are scored by the\n            LLM judge via TWO reference-free rubric criteria (RAGAS\n            ``RubricsScoreWithoutReference``): (1) Russian-language literacy,\n            (2) consistency of the generated turn with its input context. The\n            per-example result lands in\n            :attr:`ScoredMetrics.rubric_score` (the strict harmonic mean of\n            the two normalized 1–5 Likert scores). This is the anchor-side\n            companion of AC: it gives the judge FULL val-pool coverage with a\n            methodology DIFFERENT from the in-loop reward (which scores\n            reasoning by the deterministic ``reasoning_check`` + the small\n            grammar-classifier fluency signal) — protecting the ctrl-point\n            against reward hacking while keeping validation complete. ``False``\n            (default) keeps reasoning turns rubric-free (AC/SemSim behaviour\n            unchanged; ``run_eval.py`` opts in via ``--judge-rubrics``).\n        fluency_guard_threshold: optional degradation-brake transform\n            ``min(1, fluency / threshold)``. ``None`` (the default) uses the raw\n            P(grammatical) directly — with the grammar classifier the signal is\n            well-spread and acts as an **improvement** signal (the RuCoLa\n            saturation that motivated the guard is gone). A value in (0, 1]\n            re-enables the brake for ablations.\n        fluency_subtasks: a ``{task_class -> frozenset(sub_config)}`` map that\n            further restricts fluency WITHIN a task class, keyed by the loader\n            metadata ``extra["config"]``. The default\n            :data:`DEFAULT_FLUENCY_SUBTASKS` enables fluency only for the prose\n            sub-configs of ``nerel_instruct`` (definitions + normalization). A\n            class absent from the map is governed solely by ``fluency_classes``.'

    local_embedding_model: str = DEFAULT_LOCAL_EMBEDDING_MODEL
    local_embedding_device: str | None = None
    judge: OpenAIEndpoint | None = None
    compute_answer_correctness: bool = True
    known_tool_names: frozenset[str] = frozenset()
    judge_concurrency: int = 8
    embedding_concurrency: int | None = None
    judge_timeout_s: float = 120.0
    judge_max_tokens: int = 4096
    judge_max_retries: int = 2
    # --- Fluency (grammar classifier) + code_check (Python ast) ---
    local_fluency_model: str = DEFAULT_LOCAL_FLUENCY_MODEL
    local_fluency_device: str | None = None
    fluency_concurrency: int | None = None
    fluency_classes: frozenset[str] = DEFAULT_FLUENCY_CLASSES
    fluency_calibrator_path: str | None = None
    enable_fluency: bool = True
    enable_code_check: bool = True
    enable_nerel_f1: bool = True
    enable_docsem_check: bool = True
    enable_reasoning_semsim: bool = False
    enable_judge_rubrics: bool = False
    enable_schema_check: bool = False
    enable_language_guard: bool = False
    enable_rummlu_em: bool = True
    enable_rerank_em: bool = True
    format_epsilon_floor: float = 0.0
    fluency_guard_threshold: float | None = None
    known_tool_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    fluency_subtasks: dict[str, frozenset[str]] = field(
        default_factory=lambda: dict(DEFAULT_FLUENCY_SUBTASKS)
    )

    def __post_init__(self) -> None:
        if not self.local_embedding_model:
            raise ValueError(
                'local_embedding_model must be a non-empty HuggingFace model id.\nlocal_embedding_model'
            )
        if self.judge_concurrency < 1:
            raise ValueError(
                'judge_concurrency must be >= 1.\njudge_concurrency'
            )
        if self.embedding_concurrency is None:
            object.__setattr__(self, "embedding_concurrency", self.judge_concurrency)
        if self.embedding_concurrency < 1:
            raise ValueError(
                'embedding_concurrency must be >= 1.\nembedding_concurrency'
            )
        if self.judge_timeout_s <= 0:
            raise ValueError(
                'judge_timeout_s must be positive.\njudge_timeout_s'
            )
        if self.judge_max_tokens < 1:
            raise ValueError(
                'judge_max_tokens must be >= 1.\njudge_max_tokens'
            )
        if self.judge_max_retries < 0:
            raise ValueError(
                'judge_max_retries must be >= 0.\njudge_max_retries'
            )
        # Fluency concurrency defaults to the embedding limit (the fluency model
        # shares the GPU with the embedder + policy).
        
        if self.fluency_concurrency is None:
            object.__setattr__(
                self, "fluency_concurrency", self.embedding_concurrency
            )
        if self.fluency_concurrency < 1:
            raise ValueError(
                'fluency_concurrency must be >= 1.\nfluency_concurrency'
            )
        if self.enable_fluency and not self.local_fluency_model:
            raise ValueError(
                'local_fluency_model must be a non-empty HuggingFace model id when enable_fluency is True.\nlocal_fluency_model'
            )
        # Validate the fluency allowlist: known classes only (TASK_CLASSES ∪
        # EXTRACTION_LINE_TASK_CLASSES — consolidation and QA are NL tasks).
        
        unknown_fluency = set(self.fluency_classes) - set(
            TASK_CLASSES
        ) - set(EXTRACTION_LINE_TASK_CLASSES)
        if unknown_fluency:
            raise ValueError(
                f'Unknown task class(es) in fluency_classes: {sorted(unknown_fluency)}'
            )
        # Validate the sub-config allowlist map: keys must be known task classes
        # (and the fluency signal must be reachable for them — they need to be in
        # fluency_classes too, otherwise the subtask map is dead config).
        
        unknown_sub_keys = set(self.fluency_subtasks) - set(
            TASK_CLASSES
        ) - set(EXTRACTION_LINE_TASK_CLASSES)
        if unknown_sub_keys:
            raise ValueError(
                f'Unknown task class(es) in fluency_subtasks keys: {sorted(unknown_sub_keys)}'
            )
        sub_without_fluency = set(self.fluency_subtasks) - set(self.fluency_classes)
        if sub_without_fluency:
            raise ValueError(
                f'fluency_subtasks keys not in fluency_classes (dead config): {sorted(sub_without_fluency)}'
            )
        # The guard threshold must be a proper fraction of the [0, 1] scale.
        
        if self.fluency_guard_threshold is not None and not (
            0.0 < self.fluency_guard_threshold <= 1.0
        ):
            raise ValueError(
                f'fluency_guard_threshold must be in (0, 1], got {self.fluency_guard_threshold}'
            )
        # The format epsilon floor: 0.0 = off (strict), otherwise a proper
        # fraction strictly below 1 (a floor of 1.0 would make format
        # meaningless).
        if not 0.0 <= self.format_epsilon_floor < 1.0:
            raise ValueError(
                f'format_epsilon_floor must be in [0, 1), got {self.format_epsilon_floor}'
            )

    def describe(self) -> dict:
        'Log-safe dict (api keys masked'
        emb_desc = {
            "local_embedding_model": self.local_embedding_model,
            "local_embedding_device": self.local_embedding_device or "auto",
        }
        fluency_desc = {
            "local_fluency_model": self.local_fluency_model,
            "local_fluency_device": self.local_fluency_device or "auto",
            "enable_fluency": self.enable_fluency,
            "enable_code_check": self.enable_code_check,
            "enable_nerel_f1": self.enable_nerel_f1,
            "enable_docsem_check": self.enable_docsem_check,
            "enable_reasoning_semsim": self.enable_reasoning_semsim,
            "enable_schema_check": self.enable_schema_check,
            "enable_language_guard": self.enable_language_guard,
            "enable_rummlu_em": self.enable_rummlu_em,
            "enable_rerank_em": self.enable_rerank_em,
            "format_epsilon_floor": self.format_epsilon_floor,
            "fluency_guard_threshold": self.fluency_guard_threshold,
            "fluency_classes_count": len(self.fluency_classes),
            "fluency_subtasks": {
                k: sorted(v) for k, v in self.fluency_subtasks.items()
            },
            "fluency_calibrator_path": self.fluency_calibrator_path,
        }
        return {
            "judge": self.judge.describe() if self.judge is not None else None,
            "embeddings": emb_desc,
            "fluency": fluency_desc,
            "compute_answer_correctness": self.compute_answer_correctness,
            "enable_judge_rubrics": self.enable_judge_rubrics,
            "known_tool_names_count": len(self.known_tool_names),
            "known_tool_schemas_count": len(self.known_tool_schemas),
            "judge_concurrency": self.judge_concurrency,
            "embedding_concurrency": self.embedding_concurrency,
            "fluency_concurrency": self.fluency_concurrency,
            "judge_timeout_s": self.judge_timeout_s,
            "judge_max_tokens": self.judge_max_tokens,
            "judge_max_retries": self.judge_max_retries,
        }


@dataclass
class ScoredMetrics:
    'Per-example metric breakdown returned by a :class:`Scorer`.\n\n\nAttributes\n        semantic_similarity: RAGAS ``AnswerSimilarity`` ∈ [0, 1]. ``None`` for\n            the NEREL extraction sub-configs when the deterministic F1 replaces\n            it (see :attr:`RewardConfig.enable_nerel_f1`). In all other cases it\n            is computed and is the workhorse embedding-only backbone of the\n            reward. RAGAS ``AnswerSimilarity`` (``None``\n        combined: the final reward for this example — the harmonic mean of every\n            active metric (SemSim or NEREL F1; AC when the judge ran;\n            format_score for tool-use classes). A zero on any active metric\n            zeroes the whole reward\n        answer_correctness: RAGAS ``AnswerCorrectness`` ∈ [0, 1], or ``None``\n            when not computed (no judge endpoint or\n            ``compute_answer_correctness=False``). ``None`` here means "not\n            applicable" and does NOT count as a failure.\n            RAGAS ``AnswerCorrectness``\n        format_score: structural score ∈ [0, 1] for tool-use classes; ``None``\n            for non-tool-use classes (excluded from the harmonic combine).\n        fluency: fluency ∈ [0, 1] for natural-language classes (and\n            ``coder`` prose); ``None`` when not computed (class outside the\n            allowlist, no prose sentence with enough words after stripping\n            non-prose, or ``enable_fluency=False``). ``None`` here means "not\n            applicable" and does NOT count as a failure\n        code_check: structural score ∈ [0, 1] for the ``coder`` class (Python\n            ``ast`` validation); ``None`` for other classes and for ``coder``\n            answers without Python blocks. ``None`` = not applicable.\n        nerel_f1: deterministic set-F1 ∈ [0, 1] for the NEREL extraction\n            sub-configs (``entity_recognition`` / ``relation_extraction`` /\n            ``relation_keywords``), ported from the EleutherAI NEREL-bench\n            (line-set F1 over Snowball-stemmed tokens). ``None`` for everything\n            else. When active it REPLACES Semantic Similarity (SemSim is not\n            computed for these sub-configs\n        docsem_score: deterministic DocSem check ∈ [0, 1] for the\n            ``docsem_quant`` class — the strict harmonic mean of the JSON\n            format, the program-execution, the canonical answer equality and\n            the evidence set-F1 (:func:`llm_training.reward.docsem.docsem_check`;\n            the program runs in the safe AST-whitelist interpreter). ``None``\n            for everything else. When active it REPLACES Semantic Similarity\n            (SemSim is not computed for the class\n        rubric_score: judge-side rubric score ∈ [0, 1] for SGR reasoning\n            turns when ``enable_judge_rubrics`` is on and a judge is\n            configured: the strict harmonic mean of two reference-free\n            rubric criteria (Russian-language literacy; consistency with\n            the input context), each normalized from a 1–5 Likert scale. ``None``\n            everywhere else (non-reasoning turns, no judge, flag off) —\n            ``None`` means "not applicable" and does NOT count as a failure;\n            a rubric call that was attempted and failed is recorded in\n            ``failed_metrics`` as ``"rubrics"``.\n        rummlu_em: deterministic option-letter exact match ∈ {0.0, 1.0} for\n            the ``rummlu_pro`` class (the letter of the chosen option,\n            extracted from the response and compared with the reference —\n            see :attr:`RewardConfig.enable_rummlu_em`). ``None`` for\n            everything else. When active it REPLACES Semantic Similarity\n            (SemSim is not computed for the class\n        rerank_em: deterministic Meno-stand rerank grading ∈ {0.0, 0.3, 0.65,\n            1.0} for the ``meno_rerank`` class: 0.3 for a parseable JSON\n            verdict whose label (integer or string) normalizes to a valid\n            grade + 0.7 scaled by the ORDINAL distance to the distilled gold\n            verdict from ``extra["gold_label"]`` (exact 1.0; an adjacent\n            grade |Δ|=1 → 0.65; the inverted verdict 0↔2 → 0.3; see\n            :attr:`RewardConfig.enable_rerank_em`). ``None`` for everything\n            else. When active it REPLACES Semantic Similarity and doubles as\n            the anchor val\'s structural signal for the class\'s fixed val\n        schema_check: deterministic argument-schema score ∈ [0, 1] for\n            tool-CALL turns of tool-use classes when\n            :attr:`RewardConfig.enable_schema_check` is on: the mean per-call\n            grade over ``required`` presence (0.5) and declared types/enum\n            (0.5) against the tool\'s parameter JSON schema (a stdlib-only\n            subset — see :func:`llm_training.reward.schema_check.schema_check`).\n            ``None`` everywhere else (non-tool classes, text-only turns, an\n            empty schema registry, calls of schema-unknown tools) — ``None``\n            means "not applicable"\n            tool-call\n        failed_metrics: names of metrics that were **expected** but failed due\n            to a judge/embedding/fluency error. Non-applicable metrics\n            (format_score for a non-tool class; AC without a judge; fluency for\n            a short answer) do NOT count as failures.'

    semantic_similarity: float | None
    combined: float
    answer_correctness: float | None = None
    format_score: float | None = None
    fluency: float | None = None
    code_check: float | None = None
    nerel_f1: float | None = None
    docsem_score: float | None = None
    rubric_score: float | None = None
    rummlu_em: float | None = None
    schema_check: float | None = None
    rerank_em: float | None = None
    failed_metrics: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "semantic_similarity": self.semantic_similarity,
            "answer_correctness": self.answer_correctness,
            "format_score": self.format_score,
            "fluency": self.fluency,
            "code_check": self.code_check,
            "nerel_f1": self.nerel_f1,
            "docsem_score": self.docsem_score,
            "rubric_score": self.rubric_score,
            "rummlu_em": self.rummlu_em,
            "schema_check": self.schema_check,
            "rerank_em": self.rerank_em,
            "combined": self.combined,
            "failed_metrics": list(self.failed_metrics),
        }


@runtime_checkable
class Scorer(Protocol):
    'Interface for scoring one (user_input, response, reference[, task_class]) row.\n\n\n    The dispatch reward function (:mod:`llm_training.reward.dispatch`) depends on\n    this protocol, not on RAGAS, so unit tests inject a fake. The real\n    RAGAS-backed implementation is :func:`llm_training.reward.judge.make_ragas_scorer`.\n\n    The ``task_class`` parameter lets the scorer run per-class logic (currently:\n    :func:`format_check` for tool-use classes) without the caller computing it\n    out-of-band\n\n    The ``extra`` parameter carries per-example metadata (e.g.\n    ``is_call_turn`` / ``is_final_turn`` from the loaders). The scorer uses it\n    to skip ``format_check`` for text-only final-answer turns in tool-use\n    classes — without this, a correct text answer after tool calls would\n    receive ``format_score=0`` (no ``<tool_call>`` block) and zero the\n    combined reward via the harmonic mean. When ``None`` (default),\n    ``format_check`` runs unconditionally (backward-compatible).\nbackward-compatible.\n\n    The optional ``prompt`` parameter carries the ORIGINAL message-list prompt\n    (``None`` on the GRPO in-loop path, where TRL rows are pre-rendered text).\n    The judge-rubric criteria use it to rebuild the input context of a\n    reasoning turn (:func:`llm_training.reward.inputs.build_rubric_context`);\n    callers that never enable rubrics may omit it.'

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        reference: str,
        task_class: str,
        extra: dict[str, Any] | None = None,
        prompt: "list[dict] | None" = None,
    ) -> ScoredMetrics: ...
