'Tests for scripts/calibrate_fluency.py — endpoint wiring + labelling call.\n\n\nThe script lives outside the installed package, so it is loaded by path via\n``importlib``. Every test is offline: no ``openai`` client is built (the\nvalidation paths fail before that) and the labelling call is exercised with a\nfake client.'

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "calibrate_fluency.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_calibrate_fluency", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cf = _load_module()


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        label_base_url="https://api.z.ai/api/paas/v4/",
        label_model="glm-5.2",
        label_api_key="sk-test",
        label_thinking="auto",
        label_send_seed=False,
        seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _Message:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _FakeCompletions:
    'Records create() kwargs; returns a preset reply or raises.'

    def __init__(self, reply="3", fail_times=0, exc=None):
        self.reply = reply
        self.fail_times = fail_times
        self.exc = exc or RuntimeError("boom")
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.exc
        if isinstance(self.reply, _Message):
            return _Response(self.reply)
        return _Response(_Message(content=self.reply))


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("_Chat", (), {"completions": completions})()


# ---------------------------------------------------------------------------
# _is_local_endpoint
# ---------------------------------------------------------------------------
class TestIsLocalEndpoint:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8001/v1",
            "http://127.0.0.1:8001/v1",
            "http://0.0.0.0:8000/v1",
        ],
    )
    def test_local(self, url):
        assert cf._is_local_endpoint(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.z.ai/api/paas/v4/",
            "https://vsellm.ru/v1",
            "https://api.example.com/v1",
        ],
    )
    def test_remote(self, url):
        assert cf._is_local_endpoint(url) is False


# ---------------------------------------------------------------------------
# _build_label_client validation
# ---------------------------------------------------------------------------
class TestBuildLabelClientValidation:
    def test_missing_base_url_exits(self):
        with pytest.raises(SystemExit, match="endpoint is required"):
            cf._build_label_client(_args(label_base_url=None))

    def test_missing_model_exits(self):
        with pytest.raises(SystemExit, match="endpoint is required"):
            cf._build_label_client(_args(label_model=None))

    def test_full_method_url_rejected(self):
        # The Z.AI cURL docs show .../v4/chat/completions — passing that would
        # make the SDK build a doubled path. It must be rejected up front.
        
        with pytest.raises(SystemExit, match="API root"):
            cf._build_label_client(
                _args(label_base_url="https://api.z.ai/api/paas/v4/chat/completions")
            )

    def test_full_method_url_with_trailing_slash_rejected(self):
        with pytest.raises(SystemExit, match="API root"):
            cf._build_label_client(
                _args(label_base_url="https://api.z.ai/api/paas/v4/chat/completions/")
            )

    def test_remote_without_key_exits(self):
        # A remote endpoint with no key would 401 deep inside the SDK; fail early.
        
        with pytest.raises(SystemExit, match="API key is required"):
            cf._build_label_client(_args(label_api_key=None))

    def test_local_without_key_is_allowed(self, monkeypatch):
        # A local vLLM does not validate the key → the sentinel is fine.
        
        captured: dict = {}

        class _FakeOpenAI:
            def __init__(self, base_url=None, api_key=None):
                captured["base_url"] = base_url
                captured["api_key"] = api_key

        fake_mod = type("_M", (), {"OpenAI": _FakeOpenAI})
        monkeypatch.setitem(sys.modules, "openai", fake_mod)

        client, model = cf._build_label_client(
            _args(label_base_url="http://localhost:8001/v1", label_api_key=None)
        )
        assert model == "glm-5.2"
        assert captured["api_key"] == cf._LOCAL_API_KEY_SENTINEL
        assert captured["base_url"] == "http://localhost:8001/v1"

    def test_api_root_is_passed_verbatim(self, monkeypatch):
        # The SDK receives the ROOT; it appends chat/completions itself.
        
        captured: dict = {}

        class _FakeOpenAI:
            def __init__(self, base_url=None, api_key=None):
                captured["base_url"] = base_url
                captured["api_key"] = api_key

        monkeypatch.setitem(sys.modules, "openai", type("_M", (), {"OpenAI": _FakeOpenAI}))
        cf._build_label_client(_args())
        assert captured["base_url"] == "https://api.z.ai/api/paas/v4/"
        assert captured["api_key"] == "sk-test"


# ---------------------------------------------------------------------------
# _label_kwargs
# ---------------------------------------------------------------------------
class TestLabelKwargs:
    def test_auto_disables_thinking(self):
        kw = cf._label_kwargs(_args(label_thinking="auto"))
        assert kw["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_disabled_disables_thinking(self):
        kw = cf._label_kwargs(_args(label_thinking="disabled"))
        assert kw["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_enabled_leaves_provider_default(self):
        kw = cf._label_kwargs(_args(label_thinking="enabled"))
        assert "extra_body" not in kw

    def test_seed_is_opt_in(self):
        # `seed` is absent from the published Z.AI schema → not sent by default.
        
        assert "seed" not in cf._label_kwargs(_args(label_send_seed=False))

    def test_seed_sent_when_requested(self):
        kw = cf._label_kwargs(_args(label_send_seed=True, seed=7))
        assert kw["seed"] == 7


# ---------------------------------------------------------------------------
# _extract_label_text
# ---------------------------------------------------------------------------
class TestExtractLabelText:
    def test_content_preferred(self):
        resp = _Response(_Message(content=" 3 ", reasoning_content="1"))
        assert cf._extract_label_text(resp) == "3"

    def test_falls_back_to_reasoning_content(self):
        # GLM puts the chain of thought in reasoning_content; if the budget was
        # spent there, the digit is often still recoverable.
        
        resp = _Response(_Message(content="", reasoning_content="бы поставил 4"))
        assert cf._extract_label_text(resp) == "бы поставил 4"

    def test_empty_response_is_empty_string(self):
        assert cf._extract_label_text(_Response(_Message())) == ""

    def test_malformed_response_is_empty_string(self):
        assert cf._extract_label_text(object()) == ""


# ---------------------------------------------------------------------------
# _label_one
# ---------------------------------------------------------------------------
class TestLabelOne:
    def test_parses_bucket_to_unit_interval(self):
        comp = _FakeCompletions(reply="3")
        val = cf._label_one(_FakeClient(comp), "glm-5.2", "текст", extra_kwargs={})
        assert val == pytest.approx(3 / 4)

    def test_all_buckets(self):
        for bucket in range(5):
            comp = _FakeCompletions(reply=str(bucket))
            val = cf._label_one(_FakeClient(comp), "m", "t", extra_kwargs={})
            assert val == pytest.approx(bucket / 4)

    def test_extra_kwargs_forwarded(self):
        comp = _FakeCompletions(reply="2")
        extra = {"extra_body": {"thinking": {"type": "disabled"}}, "seed": 5}
        cf._label_one(_FakeClient(comp), "m", "t", extra_kwargs=extra)
        call = comp.calls[0]
        assert call["extra_body"] == {"thinking": {"type": "disabled"}}
        assert call["seed"] == 5
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == cf._LABEL_MAX_TOKENS

    def test_retries_once_without_extras(self):
        # A strict server may 400 on unknown fields → retry without them.
        
        comp = _FakeCompletions(reply="4", fail_times=1)
        extra = {"extra_body": {"thinking": {"type": "disabled"}}}
        val = cf._label_one(_FakeClient(comp), "m", "t", extra_kwargs=extra)
        assert val == pytest.approx(1.0)
        assert len(comp.calls) == 2
        assert "extra_body" in comp.calls[0]
        assert "extra_body" not in comp.calls[1]

    def test_no_retry_when_no_extras(self):
        comp = _FakeCompletions(fail_times=1)
        val = cf._label_one(_FakeClient(comp), "m", "t", extra_kwargs={})
        assert val is None
        assert len(comp.calls) == 1

    def test_both_attempts_fail_returns_none(self):
        comp = _FakeCompletions(fail_times=2)
        extra = {"extra_body": {"thinking": {"type": "disabled"}}}
        val = cf._label_one(_FakeClient(comp), "m", "t", extra_kwargs=extra)
        assert val is None
        assert len(comp.calls) == 2

    def test_unparseable_reply_is_none(self):
        comp = _FakeCompletions(reply="нет оценки")
        val = cf._label_one(_FakeClient(comp), "m", "t", extra_kwargs={})
        assert val is None

    def test_reasoning_fallback_used_when_content_empty(self):
        comp = _FakeCompletions(
            reply=_Message(content="", reasoning_content="ставлю 2")
        )
        val = cf._label_one(_FakeClient(comp), "m", "t", extra_kwargs={})
        assert val == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _load_nl_texts
# ---------------------------------------------------------------------------
class TestLoadNlTexts:
    def test_extracts_only_fluency_classes(self, tmp_path):
        report = tmp_path / "eval.json"
        report.write_text(
            '{"metrics": {"examples": ['
            '{"task_class": "menon_qa", "samples": [{"text": "A. B."}]},'
            '{"task_class": "rummlu_pro", "samples": [{"text": "42"}]}'
            ']}}',
            encoding="utf-8",
        )
        texts = cf._load_nl_texts(report)
        assert texts == ["A. B."]  # rummlu_pro is outside the allowlist

    def test_final_turns_contribute_extracted_answer(self, tmp_path):
        # The reward-equivalent population (Path A): final-answer turns
        # contribute the extracted ``answer`` field, not the raw envelope.
        
        import json as _json
        answer = "Связное объяснение решения для пользователя."
        envelope = (
            "<tool_call>\n"
            + _json.dumps(
                {"name": "finalanswertool",
                 "arguments": {"answer": answer, "status": "completed"}},
                ensure_ascii=False,
            )
            + "\n</tool_call>"
        )
        report = tmp_path / "eval.json"
        report.write_text(
            '{"metrics": {"examples": ['
            '{"task_class": "coder", '
            '"extra": {"is_final_answer": true, "is_call_turn": true}, '
            '"samples": [{"text": ' + _json.dumps(envelope) + "}]}"
            "]}}",
            encoding="utf-8",
        )
        texts = cf._load_nl_texts(report)
        assert texts == [answer]

    def test_final_turn_without_valid_call_skipped(self, tmp_path):
        report = tmp_path / "eval.json"
        report.write_text(
            '{"metrics": {"examples": ['
            '{"task_class": "coder", '
            '"extra": {"is_final_answer": true}, '
            '"samples": [{"text": "мусор без tool-call"}]}'
            ']}}',
            encoding="utf-8",
        )
        assert cf._load_nl_texts(report) == []

    def test_missing_report_gives_actionable_error(self, tmp_path):
        with pytest.raises(SystemExit, match="Report not found"):
            cf._load_nl_texts(tmp_path / "nope.json")

    def test_invalid_json_gives_actionable_error(self, tmp_path):
        report = tmp_path / "bad.json"
        report.write_text("not json at all", encoding="utf-8")
        with pytest.raises(SystemExit, match="not valid JSON"):
            cf._load_nl_texts(report)

    def test_non_object_json_rejected(self, tmp_path):
        report = tmp_path / "list.json"
        report.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(SystemExit, match="JSON object"):
            cf._load_nl_texts(report)


# ---------------------------------------------------------------------------
# main() fail-fast ordering
# ---------------------------------------------------------------------------
class TestMainFailFast:
    def test_endpoint_validated_before_loading_fluency_model(self, tmp_path, monkeypatch):
        # A bad endpoint must fail BEFORE the expensive raw-fluency pass (which
        # loads the fluency model and takes minutes). Guard:
        # raw_fluency_per_sentence must not be reached at all.
        
        report = tmp_path / "eval.json"
        report.write_text(
            '{"metrics": {"examples": [{"task_class": "menon_qa", '
            '"samples": [{"text": "A. B."}]}]}}',
            encoding="utf-8",
        )
        called: list[str] = []
        monkeypatch.setattr(
            cf, "raw_fluency_per_sentence",
            lambda *a, **k: called.append("raw") or [("s", 0.9)],
        )
        with pytest.raises(SystemExit, match="API root"):
            cf.main([
                "--report", str(report),
                "--label-base-url", "https://api.z.ai/api/paas/v4/chat/completions",
                "--label-model", "glm-5.2",
                "--label-api-key", "sk-x",
            ])
        assert called == []  # the fluency model was never loaded

    def test_missing_key_fails_before_loading_fluency_model(self, tmp_path, monkeypatch):
        report = tmp_path / "eval.json"
        report.write_text(
            '{"metrics": {"examples": [{"task_class": "menon_qa", '
            '"samples": [{"text": "A. B."}]}]}}',
            encoding="utf-8",
        )
        called: list[str] = []
        monkeypatch.setattr(
            cf, "raw_fluency_per_sentence",
            lambda *a, **k: called.append("raw") or [("s", 0.9)],
        )
        monkeypatch.delenv("LABEL_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="API key is required"):
            cf.main([
                "--report", str(report),
                "--label-base-url", "https://api.z.ai/api/paas/v4/",
                "--label-model", "glm-5.2",
            ])
        assert called == []
