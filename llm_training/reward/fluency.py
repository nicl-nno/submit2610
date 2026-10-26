"Fluency scorer — per-sentence grammar-classifier acceptability over prose.\n\n\nThe fluency signal measures how **fluent / grammatical** the natural-language\npart of an answer is. It is one of the per-class structural signals combined via\nthe reward's harmonic mean (alongside Semantic Similarity, Answer Correctness,\n``format_check`` for tool calls, and ``code_check`` for the ``coder`` class).\n\n\nPipeline (per answer)\n\n1. :func:`~llm_training.reward.scoring.split_prose_and_code` strips fenced code\n   blocks, ``<tool_call>`` / ``<tool_response>`` spans, and inline code, leaving\n   the natural-language prose\n2. ``razdel.sentenize`` splits the prose into Russian sentences, and\n   :func:`~llm_training.reward.scoring.count_words` drops any sentence with\n   fewer than :data:`~llm_training.reward.scoring.MIN_FLUENCY_WORDS` alphabetic\nthey carry no fluency signal. Single full sentences\n3. The fluency classifier (default\n   ``agentlans/snowflake-arctic-xs-grammar-classifier``, a tiny BERT grammar\n   classifier) scores the raw ``P(grammatical)`` per sentence, batched,\n   truncated to 512 tokens (the model's ``max_position_embeddings``).\n4. :func:`~llm_training.reward.scoring.aggregate_fluency` averages the\n   **raw** per-sentence scores (``None`` if no sentence survived the\nword-count gate\n\nUnlike the previous ruRoberta-RuCoLa model (whose ``P(acceptable)`` was\noverconfident and squashed near 0.98, requiring an isotonic calibrator), the\ngrammar classifier produces well-spread scores that match reality, so the\nreward uses the raw ``P(grammatical)`` directly — fluency is a genuine\n**improvement** signal, not just a degradation brake\n\nThe classifier is **always in-process** (``transformers``, sharing the policy\nGPU) — no HTTP server, no cloud. A bounded LRU sentence cache reuses repeated\nsentences (e.g. the reference answer's sentences recur across the N completions\nof a GRPO prompt). Concurrency is capped by an independent semaphore\n(``cfg.fluency_concurrency``), distinct from the embedding and judge semaphores."

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Any, Callable, Protocol, runtime_checkable

from llm_training.reward.calibration import Calibrator, load_calibrator
from llm_training.reward.config import RewardConfig
from llm_training.reward.scoring import (
    MIN_FLUENCY_WORDS,
    aggregate_fluency,
    count_words,
    split_prose_and_code,
)

logger = logging.getLogger(__name__)

#: Per-batch sentence count for the classifier forward pass.

DEFAULT_FLUENCY_BATCH_SIZE = 32

#: Tokenizer truncation cap (the grammar classifier's ``max_position_embeddings``).
# :
_FLUENCY_MAX_LENGTH = 512

#: Default target label index = "grammatical" (the positive class sits at
#: index 1 in both the grammar classifier and the legacy RuCoLa model).

_FLUENCY_LABEL_DEFAULT = 1

# : Maximum cached sentences (~each entry: 16-byte key + float).
MAX_FLUENCY_CACHE_SIZE = 10_000

#: A callable that splits text into a list of sentence strings.

SentenizeFn = Callable[[str], list[str]]

#: A callable that counts word-tokens in a sentence (see :func:`count_words`).

WordCountFn = Callable[[str], int]


@runtime_checkable
class FluencyClassifier(Protocol):
    'Synchronous batched grammar classifier'

    def classify(self, texts: list[str]) -> list[float]:
        """Return ``P(grammatical)`` ∈ [0, 1] for each input text, aligned."""
        ...


def _resolve_fluency_label(config: Any, preferred: int = _FLUENCY_LABEL_DEFAULT) -> int:
    'Resolve the index of the positive ("grammatical") class from a HF model config.\n\n\n    Both the grammar classifier (``{0: \'ungrammatical\', 1: \'grammatical\'}``) and\n    the legacy RuCoLa model (``{0: \'unacceptable\', 1: \'acceptable\'}``) put the\n    positive class at index 1, so the default ``preferred=1`` is returned\n    directly. The fallbacks make this robust to cosmetic label naming\n    (``LABEL_1`` / ``label_1`` / string keys).'
    id2label = getattr(config, "id2label", {}) or {}
    label2id = getattr(config, "label2id", {}) or {}
    if preferred in id2label:
        return preferred
    for name in ("grammatical", "acceptable", "LABEL_1", "label_1"):
        if name in label2id:
            return int(label2id[name])
    if str(preferred) in label2id:
        return int(label2id[str(preferred)])
    return preferred


class _TransformersClassifier:
    'Real in-process fluency classifier wrapping a HF sequence-classification model.\n\n\n    Lazy-imports ``torch`` / ``transformers`` so the module is importable in dev\n    environments without the ``train`` extra. The model is loaded on the chosen\n    device and shared across calls; concurrency is gated externally by the\n    scorer\'s semaphore. On CUDA the model is converted to **bf16** (faster;\n    the precision loss is immaterial for a fluency signal); fp32 stays on CPU.\n    Scoring is softmax over logits, ``[:, target_label]`` — the probability of\n    the positive ("grammatical") class.'

    def __init__(
        self,
        model_id: str,
        device: str | None,
        *,
        batch_size: int = DEFAULT_FLUENCY_BATCH_SIZE,
        target_label: int = _FLUENCY_LABEL_DEFAULT,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "The fluency scorer needs 'torch' and 'transformers'. Install "
                "the train extra: pip install -e '.[train]'."
                "\nДля fluency-оценщика нужны 'torch' и 'transformers': "
                "pip install -e '.[train]'."
            ) from e
        import numpy as np

        self._torch = torch
        self._np = np
        self._batch_size = batch_size
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(resolved_device)
        # bf16 on CUDA speeds the forward pass; the ~0.4 % relative precision
        # loss in P(grammatical) is immaterial. The grammar classifier is a
        # tiny BERT (~22M params), so the footprint is negligible either way.
        # fp32 stays on CPU (bf16 CPU ops are slow).
        
        
        
        use_bf16 = resolved_device.startswith("cuda")
        logger.info(
            "Loading fluency model %s on %s%s",
            model_id, resolved_device, " (bf16)" if use_bf16 else "",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.to(
            device=self._device, dtype=torch.bfloat16 if use_bf16 else None
        )
        self._model.eval()
        self._target_label = _resolve_fluency_label(
            self._model.config, target_label
        )

    def classify(self, texts: list[str]) -> list[float]:
        torch = self._torch
        np = self._np
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), self._batch_size):
                batch = list(texts[i : i + self._batch_size])
                enc = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=_FLUENCY_MAX_LENGTH,
                ).to(self._device)
                # Cast logits to fp32 right after the forward: numpy has NO
                # bfloat16 dtype, so `preds.cpu().numpy()` on a bf16 model's
                # output raises "Got unsupported ScalarType BFloat16"; fp32
                # also makes the softmax below more precise. The cast is over
                # a [batch, n_labels] tensor — negligible.
                
                # `preds.cpu().numpy()`
                # «Got unsupported ScalarType BFloat16»
                
                logits = self._model(**enc).logits.float()
                if logits.shape[-1] > 1:
                    preds = torch.softmax(logits, -1)[:, self._target_label]
                else:
                    preds = torch.sigmoid(logits)[:, 0]
                out.extend(preds.cpu().numpy().astype(float).tolist())
        return out


def _build_classifier(cfg: RewardConfig) -> FluencyClassifier:
    'Build the real transformers-backed classifier from ``cfg``.'
    return _TransformersClassifier(
        cfg.local_fluency_model,
        cfg.local_fluency_device,
    )


def _build_sentenize() -> SentenizeFn:
    'Build the default Russian sentence splitter (``razdel.sentenize``).'
    from razdel import sentenize as _sentenize  # type: ignore[import-not-found]

    def _fn(text: str) -> list[str]:
        return [s.text for s in _sentenize(text)]

    return _fn


class FluencyScorer:
    'Model-backed fluency scorer with a bounded LRU sentence cache.\n\n\n    Construction is cheap (no model load); the classifier / splitter are built\n    lazily on the first :meth:`score` call, unless injected via the DI seams\n    (``classifier`` / ``sentenize_fn``) — the path tests use to stay offline.\n\nArgs\n        cfg: reward configuration (fluency model / device / concurrency /\n            allowlist / sub-config allowlist / calibrator path / enable_fluency).\n        classifier: DI seam — any :class:`FluencyClassifier` (fake in tests).\n            ``None`` → the real transformers classifier is built lazily.\n        sentenize_fn: DI seam — a sentence splitter (fake in tests). ``None`` →\n            ``razdel.sentenize`` is built lazily.\n        word_count_fn: DI seam — a per-sentence word counter (fake in tests).\n            ``None`` → :func:`count_words` (``razdel.tokenize``) is used.\n        calibrator: a :class:`Calibrator`. ``None`` → resolved from\n            ``cfg.fluency_calibrator_path`` (``IdentityCalibrator`` when unset —\n            the default with the grammar classifier, whose raw scores are\n            already well-spread).\n\n    :meth:`score` returns the fluency ∈ [0, 1] for an applicable answer,\n    ``None`` when the metric is N/A (class outside the allowlist; sub-config\n    outside the allowlist when ``cfg.fluency_subtasks`` is set for the class;\n    ``enable_fluency=False``; empty prose; or no sentence with enough words\n    after stripping non-prose). It **raises** on a model failure (so the\n    caller — :class:`RAGASScorer` — can record ``"fluency"`` in\n    ``failed_metrics``, mirroring how Answer Correctness failures are'

    def __init__(
        self,
        cfg: RewardConfig,
        *,
        classifier: FluencyClassifier | None = None,
        sentenize_fn: SentenizeFn | None = None,
        word_count_fn: WordCountFn | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self._cfg = cfg
        self._sem = asyncio.Semaphore(cfg.fluency_concurrency)
        self._classifier = classifier
        self._sentenize_fn = sentenize_fn
        self._word_count_fn = word_count_fn
        self._calibrator = (
            calibrator if calibrator is not None else load_calibrator(cfg.fluency_calibrator_path)
        )
        self._cache: OrderedDict[bytes, float] = OrderedDict()

    @property
    def classifier(self) -> FluencyClassifier | None:
        'The classifier instance; ``None`` until the lazy first-score load.\n\n\n        Read-only accessor for cross-scorer weight sharing (grpo_train.py hands the\n        loaded instance to the judge-anchor scorer so the model is not'
        return self._classifier

    # --- cache helpers
    @staticmethod
    def _key(text: str) -> bytes:
        # Deterministic, collision-safe key (same rationale as CachingEmbeddings).
        
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()

    def _store(self, text: str, score: float) -> None:
        key = self._key(text)
        self._cache[key] = score
        self._cache.move_to_end(key)
        if len(self._cache) > MAX_FLUENCY_CACHE_SIZE:
            self._cache.popitem(last=False)

    def _ensure_sentenize(self) -> None:
        # Light: razdel is a small rule-based splitter. Built on first use.
        
        if self._sentenize_fn is None:
            self._sentenize_fn = _build_sentenize()

    def _ensure_word_count(self) -> None:
        # Light: count_words lazy-imports razdel only on first call. Built on
        # first use; injected in tests via the word_count_fn DI seam.
        
        if self._word_count_fn is None:
            self._word_count_fn = count_words

    def _ensure_classifier(self) -> None:
        # Heavy: loads the transformers model. Called only after the word-count
        # gate, so short answers never trigger a model load.
        
        
        if self._classifier is None:
            self._classifier = _build_classifier(self._cfg)

    async def score(
        self,
        response: str,
        task_class: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> float | None:
        'Score the fluency of ``response``; ``None`` when N/A; raise on model error.\n\n\n        ``extra`` carries per-example metadata. For a task class listed in\n        ``cfg.fluency_subtasks`` (e.g. ``nerel_instruct``), fluency runs only\n        when ``extra["config"]`` is in that sub-config allowlist — so the\n        definition / normalization sub-configs get fluency while the extraction\n        sub-configs (scored by a deterministic F1) do not.'
        if not self._cfg.enable_fluency:
            return None
        if task_class not in self._cfg.fluency_classes:
            return None
        # Sub-config gating (e.g. nerel_instruct → only definition/normalization
        # sub-configs). When the class has a subtask allowlist, the example's
        # ``extra["config"]`` must be in it.
        
        
        subtask_allowlist = self._cfg.fluency_subtasks.get(task_class)
        if subtask_allowlist is not None:
            sub = (extra or {}).get("config")
            if sub not in subtask_allowlist:
                return None
        prose, _ = split_prose_and_code(response)
        if not prose:
            return None
        # Build the light splitter + word-counter FIRST, so the heavy model is
        # never loaded for answers whose prose yields no scoreable sentence
        # (the common case for terse answers).
        
        
        self._ensure_sentenize()
        self._ensure_word_count()
        assert self._sentenize_fn is not None
        assert self._word_count_fn is not None
        raw_sentences = [s for s in self._sentenize_fn(prose) if s and s.strip()]
        # Per-sentence word-count gate: drop sentences with too few words.
        
        sentences = [
            s for s in raw_sentences if self._word_count_fn(s) >= MIN_FLUENCY_WORDS
        ]
        if not sentences:
            return None
        # Only now load the model — we have sentences to classify.
        
        self._ensure_classifier()
        per_sentence_raw = await self._score_sentences(sentences)
        # Apply the (identity by default) calibrator EACH sentence, then
        # aggregate. With the grammar classifier the calibrator is a no-op.
        
        
        calibrated = [self._calibrator.apply(r) for r in per_sentence_raw]
        return aggregate_fluency(calibrated)

    async def _score_sentences(self, sentences: list[str]) -> list[float]:
        'Classify sentences (cache hits first, then one batched call for misses).\n\n\n        The caller must have ensured the classifier is loaded (:meth:`_ensure_classifier`).'
        assert self._classifier is not None
        results: list[float | None] = [None] * len(sentences)
        miss_idx: list[int] = []
        miss_texts: list[str] = []
        for i, s in enumerate(sentences):
            key = self._key(s)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                results[i] = cached
            else:
                miss_idx.append(i)
                miss_texts.append(s)
        if miss_texts:
            # The semaphore caps concurrent in-process fluency forward passes
            # (the model shares the GPU with the embedder + policy). to_thread
            # runs the sync forward pass off the event loop.
            
            
            async with self._sem:
                scores = await asyncio.to_thread(
                    self._classifier.classify, miss_texts
                )
            for i, text, sc in zip(miss_idx, miss_texts, scores):
                results[i] = float(sc)
                self._store(text, float(sc))
        return [r for r in results if r is not None]


def make_fluency_scorer(cfg: RewardConfig) -> FluencyScorer:
    'Build the real fluency scorer (lazy model load on first :meth:`score`).'
    return FluencyScorer(cfg)


def raw_fluency_per_sentence(
    texts: list[str],
    cfg: RewardConfig,
    *,
    classifier: FluencyClassifier | None = None,
    sentenize_fn: SentenizeFn | None = None,
    word_count_fn: WordCountFn | None = None,
) -> list[tuple[str, float]]:
    'Compute RAW per-sentence ``P(grammatical)`` for offline analysis.\n\n\n    Used by the offline Phase-0 tooling — ``scripts/measure_signal_variance.py``\n    (the go/no-go distribution). The online reward path does NOT use this — it\n    goes through :class:`FluencyScorer` (cached + async).\n\n    For each text: strip code → segment → drop sentences with fewer than\n    :data:`~llm_training.reward.scoring.MIN_FLUENCY_WORDS` words. The surviving\n    sentences are **deduplicated across all texts** (many answers share\n    boilerplate) and each unique sentence is classified once. Returns\n    ``(sentence, raw_score)`` pairs (no calibration) in first-seen order.'
    clf = classifier if classifier is not None else _build_classifier(cfg)
    sent = sentenize_fn if sentenize_fn is not None else _build_sentenize()
    wcount = word_count_fn if word_count_fn is not None else count_words
    seen: dict[bytes, str] = {}
    order: list[bytes] = []
    for text in texts:
        if not text:
            continue
        prose, _ = split_prose_and_code(text)
        if not prose:
            continue
        for s in sent(prose):
            s = (s or "").strip()
            if not s or wcount(s) < MIN_FLUENCY_WORDS:
                continue
            key = hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()
            if key not in seen:
                seen[key] = s
                order.append(key)
    if not order:
        return []
    unique_sentences = [seen[k] for k in order]
    raw_scores = clf.classify(unique_sentences)
    return list(zip(unique_sentences, raw_scores))
