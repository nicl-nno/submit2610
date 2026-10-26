'ragas version compatibility: the 0.4.x judge surface on ragas 0.2.x.\n\n\nThe reward layer\'s judge (:mod:`llm_training.reward.judge`) was written\nagainst **ragas ≥ 0.4.0** — ``SemanticSimilarity`` /\n``AnswerCorrectness`` / ``RubricsScoreWithoutReference`` from\n``ragas.metrics.collections`` plus the ``llm_factory`` / ``embedding_factory``\nfactories. The ``vllm_env`` cluster stack carries **ragas 0.2.15** (frozen:\nupgrading it risks the other vllm_env consumers), where:\n\n* ``SemanticSimilarity`` / ``AnswerCorrectness`` / ``RubricsScore`` exist in\n  ``ragas.metrics`` (verified against the v0.2.15 sources) but their\n  ``ascore`` takes a **row dict / SingleTurnSample**, not the kwargs form\n  judge.py uses, and the embeddings interface is an **async ``embed_text``\n  with no ``aembed_text``**;\n* ``RubricsScoreWithoutReference`` does NOT exist (added in 0.4.0) — but the\n  0.2.15 ``RubricsScore`` already scores reference-free when its rubric texts\n  do not mention a reference and the sample carries none;\n* ``llm_factory`` builds a langchain ``ChatOpenAI`` and accepts neither\n  ``client=`` nor ``max_tokens=`` — judge.py passes its own (sanitizer-wrapped)\n  ``AsyncOpenAI`` client;\n* ``embedding_factory`` builds **OpenAI** embeddings — useless for the\n  in-process bge-m3 (the 0.4.x ``embedding_factory("huggingface", ...)`` kind\n  parameter does not exist).\n\nThis module detects the installed version and returns the SAME five names\nwith the SAME construction/call surface as 0.4.x, adapting 0.2.15 under the\nhood. On ragas ≥ 0.4.0 the native classes pass through byte-identically (the\nproduction path of grpo_train.py / run_eval.py is untouched).\n\n\nNOTE on comparability\nscored through different ragas versions yields slightly different absolute\nAC/rubric numbers (the prompt plumbing — statement-generation prompts, rubric\nscaffolds, JSON parsing — differs between 0.2.x and 0.4.x). Anchors produced\nby sft_train.py (vllm_env / ragas 0.2.15) are comparable with the GRPO series\nQUALITATIVELY; for exact numbers re-score both checkpoints with\n``run_eval.py`` on one stack.'

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: The minimum modern version — at or above it the native classes pass
#: through; below it the 0.2.x adapters are built.

_MODERN_RAGAS = (0, 4)


def ragas_version() -> tuple[int, int] | None:
    'The installed ragas version as a (major, minor) tuple (None = unknown).'
    version: str | None = None
    try:
        version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if not version:
        try:
            import ragas

            version = getattr(ragas, "__version__", None) or None
        except Exception:  # noqa: BLE001 - probing only
            return None
    if not version:
        return None
    parts = version.split(".")[:2]
    try:
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------- 0.2.x
def _build_02_metrics() -> tuple[Any, Any, Any]:
    "Build the judge-facing metric classes over ragas 0.2.x natives.\n\n\n    Returns ``(AnswerCorrectness, SemanticSimilarity,\n    RubricsScoreWithoutReference)`` — constructor signatures and ``ascore``\n    kwargs identical to the ragas 0.4.x classes judge.py was written against:\n\n    * ``SemanticSimilarity(embeddings=...)`` — the cosine recomputed on the\n      assigned embeddings object (the 0.2.15 native metric awaits a\n      synchronous-in-0.4 ``embed_text``; ours prefers ``aembed_text``, which\n      judge.py's :class:`~llm_training.reward.judge.CachingEmbeddings`\n      provides). Also accepts the row-dict form the native AnswerCorrectness\n      calls its internal similarity with.\n    * ``AnswerCorrectness(llm=..., embeddings=...)`` — wraps the native\n      metric (statement prompts + the TP/FP/FN classifier run through OUR\n      ``BaseRagasLLM`` adapter, weights [0.75, 0.25] as in 0.4.x); the\n      ``embeddings`` attribute forwards to the native metric AND its internal\n      similarity so judge.py's cache re-wrap keeps working.\n    * ``RubricsScoreWithoutReference(llm=..., rubrics=...)`` — the 0.2.15\n      ``RubricsScore`` scores reference-free when the sample carries no\n      reference and the rubric texts are ours (they never mention one); the\n      paper-grade Likert 1–5 lands in ``_normalize_rubric`` unchanged."
    from ragas.metrics import AnswerCorrectness, RubricsScore
    from ragas.run_config import RunConfig

    class _SemanticSimilarityCompat:
        'SemanticSimilarity over any embeddings object (0.4.x call form).'

        def __init__(self, embeddings: Any = None) -> None:
            self.embeddings = embeddings

        async def _embed(self, text: str) -> Any:
            emb = self.embeddings
            fn = getattr(emb, "aembed_text", None)
            if fn is not None:
                return await fn(text)
            return await emb.embed_text(text)  # 0.2.15 native (async)

        async def ascore(
            self,
            row: dict[str, Any] | None = None,
            *,
            response: str | None = None,
            reference: str | None = None,
            callbacks: Any = None,
            **kwargs: Any,
        ) -> float:
            if row is not None and response is None and reference is None:
                response = row.get("response")
                reference = row.get("reference")
            import numpy as np

            v1 = np.array(await self._embed(reference or " "))
            v2 = np.array(await self._embed(response or " "))
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            return float(v1 @ v2)

    class _AnswerCorrectnessCompat:
        'AnswerCorrectness over the native 0.2.x metric (0.4.x call form).'

        def __init__(self, llm: Any = None, embeddings: Any = None) -> None:
            self._similarity = _SemanticSimilarityCompat(embeddings=embeddings)
            self._native = AnswerCorrectness(
                llm=llm,
                embeddings=embeddings,
                answer_similarity=self._similarity,
            )
            # init() builds the internal similarity when it is None and wires
            # run configs; ours is passed pre-built, so it survives verbatim.
            self._native.init(RunConfig(max_retries=2))
            self._llm = llm
            self._embeddings = embeddings

        @property
        def embeddings(self) -> Any:
            return self._embeddings

        @embeddings.setter
        def embeddings(self, emb: Any) -> None:
            self._embeddings = emb
            self._native.embeddings = emb
            # Keep the native metric's INTERNAL similarity in sync — judge.py
            # re-wraps ``metric.embeddings`` with CachingEmbeddings after
            # construction; without the forwarding the internal calls would
            # bypass the cache and its semaphore.
            self._similarity.embeddings = emb

        @property
        def llm(self) -> Any:
            return self._llm

        async def ascore(
            self,
            *,
            user_input: str | None = None,
            response: str | None = None,
            reference: str | None = None,
            **kwargs: Any,
        ) -> float:
            from ragas.dataset_schema import SingleTurnSample

            sample = SingleTurnSample(
                user_input=user_input, response=response, reference=reference
            )
            return await self._native.single_turn_ascore(sample)

    class _RubricsScoreWithoutReferenceCompat:
        'Reference-free rubric scoring over the 0.2.x RubricsScore.'

        def __init__(self, llm: Any = None, rubrics: dict[str, str] | None = None):
            self._native = RubricsScore(rubrics=rubrics or {}, llm=llm)

        async def ascore(
            self,
            *,
            user_input: str | None = None,
            response: str | None = None,
            **kwargs: Any,
        ) -> float:
            from ragas.dataset_schema import SingleTurnSample

            sample = SingleTurnSample(user_input=user_input, response=response)
            return await self._native.single_turn_ascore(sample)

    return _AnswerCorrectnessCompat, _SemanticSimilarityCompat, (
        _RubricsScoreWithoutReferenceCompat
    )


def _build_02_factories() -> tuple[Callable[..., Any], Callable[..., Any]]:
    "Build ``(llm_factory, embedding_factory)`` over ragas 0.2.x.\n\n\n    ``llm_factory(model, client=..., max_tokens=...)`` — a minimal\n    ``BaseRagasLLM`` over judge.py's OWN ``AsyncOpenAI`` client, so the JSON\n    fence sanitizer (incl. the invalid-escape repair) and the per-request\n    timeout wiring apply verbatim (the 0.2.15 native ``llm_factory`` builds\n    a langchain ChatOpenAI and touches neither ``client`` nor\n    ``max_tokens``). The requests carry ``response_format=json_object`` —\n    the server-side JSON guarantee the 0.4.x instructor path already has.\n    The pydantic-prompt plumbing of\n    0.2.x (statement generation, the rubric scaffold) calls\n    ``BaseRagasLLM.generate`` → our ``agenerate_text`` → the client's\n    ``chat.completions.create``.\n\n    ``embedding_factory(kind, model, device=...)`` — the in-process\n    ``HuggingfaceEmbeddings`` (bge-m3 on the chosen device) wrapped in an\n    adapter adding the async ``aembed_text`` the 0.4.x-style consumers (and\n    judge.py's CachingEmbeddings) call."
    from ragas.llms.base import BaseRagasLLM
    from ragas.run_config import RunConfig

    class _OpenAICompatRagasLLM(BaseRagasLLM):
        "BaseRagasLLM over judge.py's AsyncOpenAI client (no langchain)."

        def __init__(
            self,
            model: str,
            client: Any = None,
            max_tokens: int | None = None,
            run_config: Any = None,
        ) -> None:
            # Attributes are set manually (the dataclass-generated __init__ of
            # the base varies across 0.2.x micro releases; the state it owns
            # is exactly these four fields and none of them needs its logic).
            self.model = model
            self.client = client
            self.max_tokens = max_tokens
            self.run_config = run_config or RunConfig(max_retries=2)
            self.multiple_completion_supported = False
            self.cache = None

        def _create_kwargs(
            self, prompt: Any, n: int, temperature: float | None, stop: Any
        ) -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt.to_string()}],
                "n": n,
                "temperature": temperature if temperature is not None else 1e-8,
                # Server-side JSON guarantee, mirroring the 0.4.x instructor
                # path (which requests json_object through its own wiring).
                # Every consumer of this LLM (AC's statement generation, the
                # rubric scaffold) expects a JSON reply, and a free-form judge
                # was observed copying LaTeX \(…\) notation from the judged
                # math content into its strings — an escape no strict JSON
                # parser accepts. On the V0 judge server (judge_serve_slurm.sh
                # runs VLLM_USE_V1=0 exactly for this) xgrammar enforces the
                # JSON grammar at the token level, making invalid escapes
                # unrepresentable; a V1 server without structured-output
                # support silently ignores the field (a no-op, the
                # client-side escape repair in judge.py remains the fallback).
                
                
                
                
                
                
                
                
                # structured output
                
                "response_format": {"type": "json_object"},
            }
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if stop:
                kwargs["stop"] = stop
            return kwargs

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = None,
            stop: Any = None,
            callbacks: Any = None,
        ) -> Any:
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, LLMResult

            resp = await self.client.chat.completions.create(
                **self._create_kwargs(prompt, n, temperature, stop)
            )
            generations = [
                [
                    ChatGeneration(
                        text=(choice.message.content or ""),
                        message=AIMessage(content=choice.message.content or ""),
                        generation_info={"finish_reason": choice.finish_reason},
                    )
                    for choice in resp.choices
                ]
            ]
            return LLMResult(generations=generations)

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = None,
            stop: Any = None,
            callbacks: Any = None,
        ) -> Any:
            import asyncio

            coro = self.agenerate_text(
                prompt, n=n, temperature=temperature, stop=stop,
                callbacks=callbacks,
            )
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            # A sync call inside a running loop (unused in our flows): run the
            # coroutine in a worker thread's own loop.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()

        def is_finished(self, response: Any) -> bool:
            for run in response.generations:
                for gen in run:
                    info = getattr(gen, "generation_info", None) or {}
                    reason = info.get("finish_reason")
                    if reason in ("length", "MAX_TOKENS"):
                        return False
            return True

    class _EmbeddingsAdapter:
        'Async ``aembed_text`` over the 0.2.15 native embeddings.'

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            fn = getattr(self._inner, "aembed_text", None)
            if fn is not None:
                return await fn(text)
            return await self._inner.embed_text(text)

        async def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            # The native 0.2.15 signature (async) — delegate.
            return await self.aembed_text(text)

        def set_run_config(self, run_config: Any) -> None:
            fn = getattr(self._inner, "set_run_config", None)
            if callable(fn):
                fn(run_config)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    def _llm_factory(
        model: str = "gpt-4o-mini",
        client: Any = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        return _OpenAICompatRagasLLM(
            model=model, client=client, max_tokens=max_tokens
        )

    def _embedding_factory(
        kind: str = "huggingface",
        model: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if kind != "huggingface":
            raise ValueError(
                f"The ragas 0.2.x compatibility embedding_factory supports only kind='huggingface', got {kind!r}"
            )
        import asyncio

        from ragas.embeddings import HuggingfaceEmbeddings

        # ragas 0.2.15 ships HuggingfaceEmbeddings ABSTRACT — a genuine
        # upstream bug: BaseRagasEmbeddings declares aembed_query /
        # aembed_documents @abstractmethod and the class implements only the
        # SYNCHRONOUS pair, so direct instantiation dies with "Can't
        # instantiate abstract class" (the first vllm_env anchor run). The
        # thread-offload implementations below restore it — the same
        # executor-offload design ragas itself uses for the sync path in
        # BaseRagasEmbeddings.embed_texts(is_async=False).
        
        # aembed_documents
        
        
        
        
        
        class _ConcreteHfEmbeddings(HuggingfaceEmbeddings):
            'Own ``__init__``: bypass the broken inherited pydantic one.\n\n\n            ragas 0.2.15 breaks HuggingfaceEmbeddings TWICE: besides the\n            missing ``aembed_*`` implementations (the abstract bug above),\n            ``BaseRagasEmbeddings.__get_pydantic_core_schema__`` returns an\n            IS-INSTANCE schema that replaces the normal dataclass-args\n            schema — so the pydantic-generated ``__init__`` the subclass\n            would inherit validates its own ``ArgsKwargs`` against that\n            check and dies with ``Input should be an instance of\n            HuggingfaceEmbeddings`` (the second vllm_env anchor run). A\n            custom ``__init__`` that assigns the 0.2.15 fields and calls\n            ``__post_init__`` (the SentenceTransformer load) sidesteps the\n            validator: field validation is unnecessary (we pass valid\n            types), and any LATER pydantic isinstance-check still passes —\n            we ARE a HuggingfaceEmbeddings instance.'

            def __init__(
                self,
                model_name: str = "BAAI/bge-small-en-v1.5",
                cache_folder: str | None = None,
                model_kwargs: dict[str, Any] | None = None,
                encode_kwargs: dict[str, Any] | None = None,
                cache: Any = None,
            ) -> None:
                self.model_name = model_name
                self.cache_folder = cache_folder
                self.model_kwargs = model_kwargs or {}
                self.encode_kwargs = encode_kwargs or {}
                self.cache = cache
                self.__post_init__()

            async def aembed_query(self, text: str) -> list[float]:
                return await asyncio.to_thread(self.embed_query, text)

            async def aembed_documents(
                self, texts: list[str]
            ) -> list[list[float]]:
                return await asyncio.to_thread(self.embed_documents, texts)

        model_kwargs = {"device": device} if device else {}
        inner = _ConcreteHfEmbeddings(
            model_name=model or "BAAI/bge-m3", model_kwargs=model_kwargs
        )
        inner.set_run_config(RunConfig())
        return _EmbeddingsAdapter(inner)

    return _llm_factory, _embedding_factory


# ------------------------------------------------------------------ dispatch
def import_ragas_metrics() -> tuple[Any, Any, Any]:
    'The judge-facing ``(AC, SemSim, RubricsScoreWithoutReference)`` trio.\n\n\n    On ragas ≥ 0.4.0 (or an unparseable version where the modern import\n    succeeds) the native classes are returned — the production path of\n    grpo_train.py / run_eval.py stays byte-identical. On ragas 0.2.x the\n    compatibility adapters are built. Raises a bilingual ImportError when\n    ragas is missing entirely.'
    version = ragas_version()
    if version is None or version >= _MODERN_RAGAS:
        try:
            from ragas.metrics.collections import (  # type: ignore[import-not-found]
                AnswerCorrectness,
                RubricsScoreWithoutReference,
                SemanticSimilarity,
            )

            return (
                AnswerCorrectness,
                SemanticSimilarity,
                RubricsScoreWithoutReference,
            )
        except ImportError:
            if version is not None and version >= _MODERN_RAGAS:
                raise ImportError(
                    "ragas >= 0.4.0 is installed but ragas.metrics.collections "
                    "cannot be imported — the installation is broken; "
                    "reinstall ragas."
                    "\nragas >= 0.4.0 установлен, но модуль "
                    "ragas.metrics.collections не импортируется — установка "
                    "повреждена; переустановите ragas."
                )
            logger.warning(
                "ragas version unknown and the modern metric import failed; "
                "falling back to the 0.2.x compatibility adapters."
            )
    logger.info(
        "ragas %s detected — using the 0.2.x compatibility adapters for the "
        "judge metrics.",
        ".".join(map(str, version)) if version else "?",
    )
    return _build_02_metrics()


def import_ragas_factories() -> tuple[Callable[..., Any], Callable[..., Any]]:
    'The judge-facing ``(llm_factory, embedding_factory)`` pair.\n\n\n    Native on ragas ≥ 0.4.0; the 0.2.x adapters otherwise (see\n    :func:`_build_02_factories`).'
    version = ragas_version()
    if version is None or version >= _MODERN_RAGAS:
        try:
            from ragas.embeddings.base import (
                embedding_factory,  # type: ignore[import-not-found]
            )
            from ragas.llms import llm_factory  # type: ignore[import-not-found]

            return llm_factory, embedding_factory
        except ImportError:
            if version is not None and version >= _MODERN_RAGAS:
                raise ImportError(
                    "ragas >= 0.4.0 is installed but its llm/embedding "
                    "factories cannot be imported — the installation is "
                    "broken; reinstall ragas."
                    "\nragas >= 0.4.0 установлен, но его фабрики llm/"
                    "эмбеддингов не импортируются — установка повреждена; "
                    "переустановите ragas."
                )
            logger.warning(
                "ragas version unknown and the modern factory import failed; "
                "falling back to the 0.2.x compatibility adapters."
            )
    logger.info(
        "ragas %s detected — using the 0.2.x compatibility adapters for the "
        "judge LLM/embedding factories.",
        ".".join(map(str, version)) if version else "?",
    )
    return _build_02_factories()
