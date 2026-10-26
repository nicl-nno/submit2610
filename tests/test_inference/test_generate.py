'Tests for inference/generate.py — PolicyGenerator (fake vLLM, no GPU).'

from __future__ import annotations

import pytest

from llm_training.inference.generate import (
    GenerationConfig,
    PolicyGenerator,
    render_prompt,
)
from llm_training.schema import UnifiedExample


class _FakeOutput:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRequestOutput:
    def __init__(self, samples: list[str]) -> None:
        self.outputs = [_FakeOutput(t) for t in samples]


class _FakeLLM:
    'Records generate() calls; returns canned samples per prompt index.'

    def __init__(self, tokenizer, samples_per_prompt=None, n_always=1) -> None:
        self.tokenizer = tokenizer
        self.calls = []
        self._samples = samples_per_prompt or []
        self._n = n_always

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params):
        self.calls.append({"prompts": list(prompts), "sampling_params": sampling_params})
        out = []
        for i, _p in enumerate(prompts):
            if i < len(self._samples):
                out.append(_FakeRequestOutput(self._samples[i]))
            else:
                out.append(_FakeRequestOutput([f"gen-{i}"] * self._n))
        return out


def _ex(task_class="coder", prompt=None, tools=None, reference="r"):
    return UnifiedExample(
        task_class=task_class,
        prompt=prompt or [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        reference=reference,
        tools=tools,
        reference_messages=[{"role": "assistant", "content": reference}],
    )


class TestGenerationConfig:
    def test_defaults_greedy(self):
        c = GenerationConfig()
        assert c.mode == "greedy"
        assert c.num_samples == 1

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            GenerationConfig(mode="weird")

    def test_greedy_forbids_multiple_samples(self):
        with pytest.raises(ValueError):
            GenerationConfig(mode="greedy", num_samples=3)

    def test_sampling_allows_multiple(self):
        c = GenerationConfig(mode="sampling", num_samples=4)
        assert c.num_samples == 4


class TestRenderPrompt:
    def test_renders_with_generation_prompt(self, tokenizer):
        ex = _ex()
        text = render_prompt(ex, tokenizer)
        # add_generation_prompt=True appends the assistant im_start marker.
        # add_generation_prompt=True
        assert "<|im_start|>assistant" in text
        assert text.rstrip().endswith("<|im_start|>assistant\n") or text.rstrip().endswith(
            "<|im_start|>assistant"
        )

    def test_tools_baked_into_system(self, tokenizer):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        ex = _ex(
            prompt=[{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
            tools=tools,
        )
        text = render_prompt(ex, tokenizer)
        assert "<tools>" in text


class TestGenerate:
    def test_greedy_returns_one_sample_per_example(self, tokenizer):
        llm = _FakeLLM(tokenizer, samples_per_prompt=[["only"], ["one"]])
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=32768)
        out = gen.generate([_ex(), _ex()], GenerationConfig(mode="greedy"))
        assert out == [["only"], ["one"]]
        # Confirm temperature=0 was used
        sp = llm.calls[0]["sampling_params"][0]
        assert sp.temperature == 0.0

    def test_sampling_returns_n_samples(self, tokenizer):
        llm = _FakeLLM(tokenizer, n_always=3)
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=32768)
        out = gen.generate(
            [_ex(), _ex()],
            GenerationConfig(mode="sampling", num_samples=3, temperature=0.8),
        )
        assert len(out) == 2
        assert all(len(samples) == 3 for samples in out)
        sp = llm.calls[0]["sampling_params"][0]
        assert sp.n == 3
        assert sp.temperature == 0.8

    def test_empty_examples_returns_empty(self, tokenizer):
        gen = PolicyGenerator(llm=_FakeLLM(tokenizer), tokenizer=tokenizer)
        assert gen.generate([], GenerationConfig()) == []

    def test_default_config_is_greedy(self, tokenizer):
        llm = _FakeLLM(tokenizer, samples_per_prompt=[["x"]])
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=32768)
        out = gen.generate([_ex()])
        assert out == [["x"]]
        assert llm.calls[0]["sampling_params"][0].temperature == 0.0

    def test_prompts_are_strings(self, tokenizer):
        llm = _FakeLLM(tokenizer, samples_per_prompt=[["a"], ["b"]])
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=32768)
        gen.generate([_ex(), _ex()])
        prompts = llm.calls[0]["prompts"]
        assert all(isinstance(p, str) for p in prompts)
        assert all("<|im_start|>assistant" in p for p in prompts)

    def test_requires_model_or_llm(self):
        with pytest.raises(ValueError):
            PolicyGenerator()


# ---------------------------------------------------------------------------
# Special-token-preserving sampling
# ---------------------------------------------------------------------------
class TestSpecialTokenDecoding:
    "The ``<tool_call>`` envelope must survive into ``out.text``.\n\n\n    vLLM's default ``skip_special_tokens=True`` strips added special tokens —\n    and ``<tool_call>``/``</tool_call>`` ARE added special tokens of the\n    Meno-Lite tokenizer — so every tool-use generation would fail\n    ``format_check`` regardless of model quality. Both flags must be pinned\n    False in both modes."

    def test_greedy_keeps_special_tokens(self, tokenizer):
        llm = _FakeLLM(tokenizer)
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=32768)
        gen.generate([_ex()], GenerationConfig(mode="greedy"))
        sp = llm.calls[0]["sampling_params"][0]
        assert sp.skip_special_tokens is False
        assert sp.spaces_between_special_tokens is False

    def test_sampling_keeps_special_tokens(self, tokenizer):
        llm = _FakeLLM(tokenizer)
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=32768)
        gen.generate([_ex()], GenerationConfig(mode="sampling", num_samples=2))
        sp = llm.calls[0]["sampling_params"][0]
        assert sp.skip_special_tokens is False
        assert sp.spaces_between_special_tokens is False


# ---------------------------------------------------------------------------
# Per-example max_tokens budget
# ---------------------------------------------------------------------------
class TestPerExampleMaxTokens:
    'The regression suite for the vLLM-16-token-default bug.'

    @staticmethod
    def _prompt_token_count(gen, text: str) -> int:
        return len(gen._tokenizer(text, add_special_tokens=False)["input_ids"])

    def test_regression_auto_never_vllm_default_16(self, tokenizer):
        # The bug: max_tokens=None used to fall through to vLLM's own default
        # (16 tokens), silently truncating every completion mid-sentence. The
        # per-example budget must be an explicit, much larger value.
        
        
        llm = _FakeLLM(tokenizer)
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=32768)
        gen.generate([_ex()], GenerationConfig())
        sp = llm.calls[0]["sampling_params"][0]
        assert sp.max_tokens != 16
        assert sp.max_tokens == min(8192, 32768 - self._prompt_token_count(
            gen, llm.calls[0]["prompts"][0]
        ))

    def test_per_example_limits_differ_by_prompt_length(self, tokenizer):
        # A short and a long prompt (near the context limit) must receive
        # DIFFERENT max_tokens — one SamplingParams entry per prompt.
        
        
        llm = _FakeLLM(tokenizer)
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=2048)
        short_ex = _ex(prompt=[{"role": "system", "content": "s"},
                               {"role": "user", "content": "короткий вопрос"}])
        long_ex = _ex(prompt=[{"role": "system", "content": "s"},
                              {"role": "user", "content": "слово " * 900}])
        gen.generate([short_ex, long_ex], GenerationConfig())
        params = llm.calls[0]["sampling_params"]
        assert isinstance(params, list) and len(params) == 2
        short_prompt_tokens = self._prompt_token_count(gen, llm.calls[0]["prompts"][0])
        long_prompt_tokens = self._prompt_token_count(gen, llm.calls[0]["prompts"][1])
        assert params[0].max_tokens == min(8192, 2048 - short_prompt_tokens)
        assert params[1].max_tokens == min(8192, 2048 - long_prompt_tokens)
        assert params[0].max_tokens > params[1].max_tokens

    def test_explicit_max_tokens_is_upper_bound(self, tokenizer):
        # An explicit --max-tokens caps the budget but is still clamped to the
        # remaining context.
        
        llm = _FakeLLM(tokenizer)
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=1024)
        gen.generate([_ex()], GenerationConfig(max_tokens=512))
        assert llm.calls[0]["sampling_params"][0].max_tokens == 512
        # A cap larger than the remaining context clamps to the remainder.
        
        gen.generate([_ex(prompt=[{"role": "system", "content": "s"},
                                   {"role": "user", "content": "слово " * 400}])],
                     GenerationConfig(max_tokens=2000))
        prompt_tokens = self._prompt_token_count(gen, llm.calls[1]["prompts"][0])
        assert llm.calls[1]["sampling_params"][0].max_tokens == 1024 - prompt_tokens

    def test_prompt_filling_context_clamps_to_one(self, tokenizer):
        # A prompt that (impossibly) leaves no room must clamp to 1, not raise.
        
        llm = _FakeLLM(tokenizer)
        gen = PolicyGenerator(llm=llm, tokenizer=tokenizer, max_model_len=128)
        huge_ex = _ex(prompt=[{"role": "system", "content": "s"},
                              {"role": "user", "content": "слово " * 200}])
        gen.generate([huge_ex], GenerationConfig())
        assert llm.calls[0]["sampling_params"][0].max_tokens == 1

    def test_unknown_max_model_len_auto_mode_raises(self, tokenizer):
        # Without any max_model_len (neither explicit nor resolvable from the
        # LLM) the auto budget is undefined — a clear error, never vLLM's 16.
        
        gen = PolicyGenerator(llm=_FakeLLM(tokenizer), tokenizer=tokenizer)
        assert gen._max_model_len is None
        with pytest.raises(RuntimeError, match="max_model_len"):
            gen.generate([_ex()], GenerationConfig())

    def test_max_model_len_resolved_from_engine_config(self, tokenizer):
        # Best-effort: llm.llm_engine.model_config.max_model_len (the vLLM 0.22
        # layout) is picked up automatically.
        # Best-effort: llm.llm_engine.model_config.max_model_len (vLLM 0.22).

        class _Cfg:
            max_model_len = 4096

        class _Engine:
            model_config = _Cfg()

        class _LLMWithEngine(_FakeLLM):
            def __init__(self):
                super().__init__(tokenizer)
                self.llm_engine = _Engine()

        gen = PolicyGenerator(llm=_LLMWithEngine(), tokenizer=tokenizer)
        assert gen._max_model_len == 4096
        llm = gen._llm
        gen.generate([_ex()], GenerationConfig())
        prompt_tokens = self._prompt_token_count(gen, llm.calls[0]["prompts"][0])
        assert llm.calls[0]["sampling_params"][0].max_tokens == min(8192, 4096 - prompt_tokens)

    def test_auto_ceiling_constant(self):
        # The auto ceiling stays at 8192 (covers the 8 000-token reference cap
        # plus the tool-call wrapper fields).
        
        from llm_training.inference.generate import AUTO_MAX_NEW_TOKENS
        assert AUTO_MAX_NEW_TOKENS == 8192
