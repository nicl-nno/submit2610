'Tests for the ragas version-compatibility layer (offline, fakes).\n\n\nThe dev environment carries ragas 0.4.3 — the native pass-through path is\nexercised against the real library. The 0.2.15 branch (the frozen vllm_env\nstack) is exercised against FAKE ragas modules injected into ``sys.modules``\nmirroring the v0.2.15 source surface pinned in\n``llm_training/reward/_ragas_compat.py`` (row-dict ``ascore``, async\n``embed_text`` with no ``aembed_text``, ``RubricsScore``-with-reference,\nfactory signatures).'

from __future__ import annotations

import abc
import sys
import types
from types import SimpleNamespace

import pytest

from llm_training.reward import _ragas_compat as rc


# --------------------------------------------------------------- version
class TestVersionParsing:
    def test_real_dev_version_is_modern(self):
        # Dev: ragas 0.4.3 → (0, 4) — the native branch.
        assert rc.ragas_version() == (0, 4)

    def test_parse_0215(self, monkeypatch):
        import importlib.metadata as im

        monkeypatch.setattr(im, "version", lambda name: "0.2.15")
        assert rc.ragas_version() == (0, 2)

    def test_parse_failure_returns_none(self, monkeypatch):
        import importlib.metadata as im

        monkeypatch.setattr(im, "version", lambda name: "unknown-version")
        monkeypatch.setattr(
            sys, "modules", {**sys.modules, "ragas": None}
        )  # force the import fallback to fail
        assert rc.ragas_version() is None


class TestNativePassthrough:
    def test_modern_version_returns_native_classes(self):
        ac, ss, rub = rc.import_ragas_metrics()

        assert ac.__module__.startswith("ragas")
        assert ss.__name__ == "SemanticSimilarity"
        assert "ragas" in rub.__module__

    def test_modern_factories_are_native(self):
        lf, ef = rc.import_ragas_factories()
        assert "ragas" in lf.__module__
        assert "ragas" in ef.__module__


# --------------------------------------------------- fake ragas 0.2.15
class _FakeRunConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeSingleTurnSample:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeNativeAnswerCorrectness:
    'The 0.2.15 AnswerCorrectness surface the compat wrapper relies on.'

    def __init__(self, llm=None, embeddings=None, answer_similarity=None):
        assert llm is not None, "MetricWithLLM.init asserts the llm"
        self.llm = llm
        self.embeddings = embeddings
        self.answer_similarity = answer_similarity
        self.initialized = False
        self.seen_samples = []

    def init(self, run_config):
        self.initialized = True

    async def single_turn_ascore(self, sample):
        self.seen_samples.append(sample)
        # The native metric calls its INTERNAL similarity with a ROW DICT
        # (the deprecated ascore form) — the compat similarity must accept it.
        
        
        sim = await self.answer_similarity.ascore(
            {"response": sample.response, "reference": sample.reference}
        )
        return 0.5 * sim + 0.25


class _FakeNativeRubricsScore:
    def __init__(self, rubrics=None, llm=None):
        self.rubrics = rubrics
        self.llm = llm
        self.seen_samples = []

    async def single_turn_ascore(self, sample):
        self.seen_samples.append(sample)
        return 4  # a Likert 1–5


class _FakeEmbeddings:
    """0.2.15-style: ASYNC embed_text, no aembed_text."""

    def __init__(self, vectors=None):
        self._vectors = vectors or {}
        self.calls = []

    async def embed_text(self, text, is_async=True):
        self.calls.append(text)
        return self._vectors[text]

    def set_run_config(self, run_config):
        self.run_config = run_config


class _FakeBaseRagasEmbeddings(abc.ABC):
    'The v0.2.15 ``BaseRagasEmbeddings`` surface (source-verified).\n\n\n    ABC declaring ``aembed_query`` / ``aembed_documents`` @abstractmethod;\n    the async ``embed_text`` / ``embed_texts`` pair lives on the BASE and\n    routes to ``aembed_documents`` (the ``is_async=False`` executor path is\n    omitted — the compat layer never takes it).'

    def __init__(self, cache=None):
        self.cache = cache
        self.run_config = None

    async def embed_text(self, text, is_async=True):
        embs = await self.embed_texts([text], is_async=is_async)
        return embs[0]

    async def embed_texts(self, texts, is_async=True):
        return await self.aembed_documents(texts)

    @abc.abstractmethod
    async def aembed_query(self, text):
        ...

    @abc.abstractmethod
    async def aembed_documents(self, texts):
        ...

    def set_run_config(self, run_config):
        self.run_config = run_config


class _FakeHuggingfaceEmbeddings(_FakeBaseRagasEmbeddings):
    'The REAL 0.2.15 HuggingfaceEmbeddings shape (the source-verified bug).\n\nthe inherited\n    ``aembed_query`` / ``aembed_documents`` stay abstract, so DIRECT\n    instantiation raises TypeError (exactly how the first vllm_env anchor\n    run died; the former concrete fake hid this from the test suite).\n\n    The class is ALSO a pydantic dataclass whose base defines\n    ``__get_pydantic_core_schema__`` as an IS-INSTANCE schema: the generated\n    ``__init__`` validates its own ``ArgsKwargs`` against it, so ANY\n    subclass INHERITING that ``__init__`` dies with a validation error\n    (the second vllm_env anchor run, «Input should be an instance of\n    HuggingfaceEmbeddings»). Simulated here: the fake ``__init__`` raises\n    for subclass instances (the parent itself is abstract and is never\n    constructed directly). ``__post_init__`` (the SentenceTransformer\n    load) is modelled with a flag + the fake vector store.'

    def __init__(self, model_name=None, cache_folder=None,
                 model_kwargs=None, encode_kwargs=None, cache=None):
        if type(self) is not _FakeHuggingfaceEmbeddings:
            raise TypeError(
                "simulated pydantic is-instance __init__ validation: a "
                "subclass must define its own __init__ (see "
                "_ragas_compat._embedding_factory) / симуляция "
                "pydantic-валидации __init__: подкласс обязан определить "
                "собственный __init__"
            )
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.cache = cache
        self.__post_init__()

    def __post_init__(self):
        # The real one calls super().__init__(cache=...) and loads the
        # SentenceTransformer; the fake records that it ran.
        
        
        super().__init__(cache=self.cache)
        self.post_init_called = True
        self._vectors = {}

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_documents(self, texts):
        return [self._vectors[t] for t in texts]


def _install_fake_ragas(monkeypatch, native_ac=_FakeNativeAnswerCorrectness):
    'Mirror the v0.2.15 module surface the compat layer imports.'
    fake_metrics = types.ModuleType("ragas.metrics")
    fake_metrics.AnswerCorrectness = native_ac
    fake_metrics.RubricsScore = _FakeNativeRubricsScore
    fake_ds = types.ModuleType("ragas.dataset_schema")
    fake_ds.SingleTurnSample = _FakeSingleTurnSample
    fake_rc_mod = types.ModuleType("ragas.run_config")
    fake_rc_mod.RunConfig = _FakeRunConfig
    fake_llms_base = types.ModuleType("ragas.llms.base")

    class _FakeBaseRagasLLM:  # the real one provides generate()/retry glue
        pass

    fake_llms_base.BaseRagasLLM = _FakeBaseRagasLLM
    fake_embeddings = types.ModuleType("ragas.embeddings")
    fake_embeddings.HuggingfaceEmbeddings = _FakeHuggingfaceEmbeddings
    fake_embeddings.BaseRagasEmbeddings = _FakeBaseRagasEmbeddings

    for name, module in [
        ("ragas.metrics", fake_metrics),
        ("ragas.dataset_schema", fake_ds),
        ("ragas.run_config", fake_rc_mod),
        ("ragas.llms.base", fake_llms_base),
        ("ragas.embeddings", fake_embeddings),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(rc, "ragas_version", lambda: (0, 2))
    return SimpleNamespace(metrics=fake_metrics)


@pytest.fixture
def fake_ragas(monkeypatch):
    return _install_fake_ragas(monkeypatch)


class TestCompatMetrics0215:
    def test_dispatch_builds_compat_trio(self, fake_ragas):
        ac_cls, ss_cls, rub_cls = rc.import_ragas_metrics()
        assert ac_cls is not None and ss_cls is not None and rub_cls is not None

    @pytest.mark.asyncio
    async def test_semantic_similarity_cosine(self, fake_ragas):
        _ac, ss_cls, _rub = rc.import_ragas_metrics()
        emb = _FakeEmbeddings({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        metric = ss_cls(embeddings=emb)
        # Orthogonal → 0; identical → 1.
        assert await metric.ascore(response="a", reference="b") == (
            pytest.approx(0.0)
        )
        assert await metric.ascore(response="a", reference="a") == (
            pytest.approx(1.0)
        )

    @pytest.mark.asyncio
    async def test_semantic_similarity_accepts_row_dict(self, fake_ragas):
        # The native AC calls its internal similarity with a row dict.
        
        _ac, ss_cls, _rub = rc.import_ragas_metrics()
        emb = _FakeEmbeddings({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        metric = ss_cls(embeddings=emb)
        assert await metric.ascore({"response": "a", "reference": "b"}) == (
            pytest.approx(0.0)
        )

    @pytest.mark.asyncio
    async def test_semantic_similarity_prefers_aembed_text(self, fake_ragas):
        # judge.py wraps the embeddings in CachingEmbeddings (async
        # aembed_text); when present it must win over the 0.2.15 embed_text.
        
        _ac, ss_cls, _rub = rc.import_ragas_metrics()

        class _Cached:
            def __init__(self, inner):
                self._inner = inner
                self.aembed_calls = 0

            async def aembed_text(self, text, **kw):
                self.aembed_calls += 1
                return await self._inner.embed_text(text)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        inner = _FakeEmbeddings({"a": [1.0, 0.0]})
        cached = _Cached(inner)
        metric = ss_cls(embeddings=cached)
        await metric.ascore(response="a", reference="a")
        assert cached.aembed_calls == 2

    @pytest.mark.asyncio
    async def test_answer_correctness_flow(self, fake_ragas):
        ac_cls, _ss_cls, _rub = rc.import_ragas_metrics()
        emb = _FakeEmbeddings({"x": [1.0, 0.0], "y": [1.0, 0.0]})
        llm = object()
        metric = ac_cls(llm=llm, embeddings=emb)
        score = await metric.ascore(
            user_input="q", response="x", reference="y"
        )
        # 0.5 * 1.0 (identical vectors) + 0.25.
        
        assert score == pytest.approx(0.75)
        native = metric._native
        assert native.initialized
        assert native.seen_samples[-1].user_input == "q"

    def test_answer_correctness_embeddings_forwarding(self, fake_ragas):
        # judge.py re-wraps metric.embeddings with CachingEmbeddings AFTER
        # construction — the setter must forward to the native metric AND its
        # internal similarity.
        # judge.py
        
        ac_cls, _ss, _rub = rc.import_ragas_metrics()
        emb1, emb2 = object(), object()
        metric = ac_cls(llm=object(), embeddings=emb1)
        assert metric.embeddings is emb1
        assert metric._native.embeddings is emb1
        assert metric._native.answer_similarity.embeddings is emb1
        metric.embeddings = emb2
        assert metric.embeddings is emb2
        assert metric._native.embeddings is emb2
        assert metric._native.answer_similarity.embeddings is emb2

    @pytest.mark.asyncio
    async def test_rubrics_without_reference_flow(self, fake_ragas):
        _ac, _ss, rub_cls = rc.import_ragas_metrics()
        llm = object()
        rubrics = {"score1_description": "плохо", "score5_description": "хорошо"}
        metric = rub_cls(llm=llm, rubrics=rubrics)
        assert await metric.ascore(user_input="задача", response="ответ") == 4
        assert metric._native.rubrics == rubrics
        # No reference field is sent (reference-free scoring).
        
        sample = metric._native.seen_samples[-1]
        assert not hasattr(sample, "reference") or sample.reference is None


class _FakeCompletions:
    def __init__(self, content='{"score": 4, "feedback": "ok"}'):
        self.content = content
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                    finish_reason="stop",
                )
            ]
        )


class TestCompatFactories0215:
    def test_llm_factory_uses_the_given_client(self, fake_ragas):
        lf, _ef = rc.import_ragas_factories()
        client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
        llm = lf("judge-model", client=client, max_tokens=4096)
        assert llm.model == "judge-model"
        assert llm.client is client
        assert llm.max_tokens == 4096
        assert llm.multiple_completion_supported is False

    @pytest.mark.asyncio
    async def test_llm_agenerate_text_roundtrip(self, fake_ragas):
        lf, _ef = rc.import_ragas_factories()
        completions = _FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        llm = lf("judge-model", client=client, max_tokens=128)

        class _Prompt:
            def to_string(self):
                return "ПРОМПТ"

        result = await llm.agenerate_text(_Prompt(), n=1, temperature=None)
        assert result.generations[0][0].text == '{"score": 4, "feedback": "ok"}'
        call = completions.calls[0]
        assert call["model"] == "judge-model"
        assert call["max_tokens"] == 128
        assert call["messages"] == [{"role": "user", "content": "ПРОМПТ"}]
        assert call["temperature"] == pytest.approx(1e-8)  # deterministic

    @pytest.mark.asyncio
    async def test_llm_requests_json_object(self, fake_ragas):
        # Every request carries response_format=json_object — the server-side
        # JSON guarantee the 0.4.x instructor path already has. On the V0
        # judge server xgrammar enforces valid escapes at the token level
        # (the LaTeX \( failure class of the 2026-09-25 SFT anchors becomes
        # unrepresentable); a server without structured output ignores the
        # field and the client-side escape repair remains the fallback.
        
        
        
        
        
        
        lf, _ef = rc.import_ragas_factories()
        completions = _FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        llm = lf("judge-model", client=client, max_tokens=128)

        class _Prompt:
            def to_string(self):
                return "ПРОМПТ"

        await llm.agenerate_text(_Prompt(), n=1, temperature=None)
        assert completions.calls[0]["response_format"] == {
            "type": "json_object"
        }

    def test_llm_is_finished_detects_truncation(self, fake_ragas):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        lf, _ef = rc.import_ragas_factories()
        llm = lf("m", client=SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions())))

        def _result(reason):
            return LLMResult(generations=[[
                ChatGeneration(
                    text="x",
                    message=AIMessage(content="x"),
                    generation_info={"finish_reason": reason},
                )
            ]])

        assert llm.is_finished(_result("stop")) is True
        assert llm.is_finished(_result("length")) is False

    def test_embedding_factory_builds_huggingface(self, fake_ragas):
        _lf, ef = rc.import_ragas_factories()
        emb = ef("huggingface", "BAAI/bge-m3", device="cpu")
        assert isinstance(emb._inner, _FakeHuggingfaceEmbeddings)
        assert emb._inner.model_name == "BAAI/bge-m3"
        assert emb._inner.model_kwargs == {"device": "cpu"}
        assert emb._inner.run_config is not None
        # The custom __init__ must run the model-load post-init (the fake's
        # flag) — otherwise the adapter wraps an empty shell.
        
        
        assert emb._inner.post_init_called is True

    def test_native_huggingface_embeddings_is_abstract_0215(self, fake_ragas):
        # Regression documentation: ragas 0.2.15 ships HuggingfaceEmbeddings
        # ABSTRACT (aembed_query/aembed_documents never implemented) — the
        # faithful fake reproduces the upstream bug; direct instantiation
        # raises TypeError exactly like the first vllm_env anchor run did.
        
        
        
        with pytest.raises(TypeError, match="abstract"):
            _FakeHuggingfaceEmbeddings(model_name="x", model_kwargs={})

    def test_naive_subclass_hits_pydantic_init_trap(self, fake_ragas):
        # The SECOND vllm_env failure layer, reproduced on the faithful
        # fake: a subclass INHERITING the parent's pydantic-generated
        # __init__ dies inside the is-instance init validation («Input
        # should be an instance of HuggingfaceEmbeddings») — which is
        # exactly why the compat subclass defines its OWN __init__.
        
        
        # is-instance
        
        class _Naive(_FakeHuggingfaceEmbeddings):
            async def aembed_query(self, text):
                return [0.0]

            async def aembed_documents(self, texts):
                return [[0.0] for _ in texts]

        with pytest.raises(TypeError, match="simulated pydantic"):
            _Naive(model_name="x", model_kwargs={})

    def test_embedding_factory_completes_the_abstract_native(self, fake_ragas):
        # The compat factory must SUBCLASS the abstract native and complete
        # the missing async pair — without the fix this constructor call
        # itself dies with "Can't instantiate abstract class".
        
        
        
        _lf, ef = rc.import_ragas_factories()
        emb = ef("huggingface", "BAAI/bge-m3", device="cpu")
        import inspect

        assert inspect.iscoroutinefunction(emb._inner.aembed_query)
        assert inspect.iscoroutinefunction(emb._inner.aembed_documents)

    def test_embedding_factory_rejects_other_kinds(self, fake_ragas):
        _lf, ef = rc.import_ragas_factories()
        with pytest.raises(ValueError, match="huggingface"):
            ef("openai", "text-embedding-ada-002")

    @pytest.mark.asyncio
    async def test_embedding_adapter_adds_aembed_text(self, fake_ragas):
        _lf, ef = rc.import_ragas_factories()
        emb = ef("huggingface", "BAAI/bge-m3", device="cpu")
        emb._inner._vectors = {"текст": [1.0, 2.0]}
        # aembed_text (the 0.4.x / CachingEmbeddings entry) delegates to the
        # native async embed_text.
        assert await emb.aembed_text("текст") == [1.0, 2.0]
        # The native async signature also stays callable.
        assert await emb.embed_text("текст") == [1.0, 2.0]


class TestJudgeWiring:
    def test_judge_importers_delegate(self):
        # The judge.py wrappers delegate to the compat layer unchanged.
        
        from llm_training.reward.judge import (
            _import_ragas_factories,
            _import_ragas_metrics,
        )

        ac, ss, rub = _import_ragas_metrics()
        lf, ef = _import_ragas_factories()
        assert (ac, ss, rub) == rc.import_ragas_metrics()
        assert (lf, ef) == rc.import_ragas_factories()
