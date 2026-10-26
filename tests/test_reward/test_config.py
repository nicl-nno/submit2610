'Tests for reward/config.py — RewardConfig validation + ScoredMetrics.\n\n\nCovers the new fields added for judge resilience / failure reporting and the\nrevised ScoredMetrics shape (Semantic Similarity always; AC and format_score\noptional; failed_metrics tracks expected-but-failed metrics only).'

from __future__ import annotations

import pytest

from llm_training.reward.config import RewardConfig, ScoredMetrics
from llm_training.runtime.endpoints import OpenAIEndpoint


def _judge() -> OpenAIEndpoint:
    return OpenAIEndpoint(base_url="http://judge/v1", model="j")


class TestRewardConfigValidation:
    def test_defaults_without_judge(self):
        # Minimal config: in-process embeddings only (no judge). This is the
        # in-loop / online reward shape for GRPO trainings. The embedding model
        # defaults to bge-m3.
        
        cfg = RewardConfig()
        assert cfg.judge is None
        assert cfg.local_embedding_model == "BAAI/bge-m3"
        assert cfg.compute_answer_correctness is True
        assert cfg.known_tool_names == frozenset()
        assert cfg.judge_max_tokens == 4096
        assert cfg.judge_max_retries == 2

    def test_with_judge(self):
        cfg = RewardConfig(judge=_judge())
        assert cfg.judge is not None
        assert cfg.judge.model == "j"

    def test_judge_max_tokens_zero_rejected(self):
        with pytest.raises(ValueError):
            RewardConfig(judge=_judge(), judge_max_tokens=0)

    def test_judge_max_tokens_negative_rejected(self):
        with pytest.raises(ValueError):
            RewardConfig(judge=_judge(), judge_max_tokens=-1)

    def test_judge_max_retries_zero_allowed(self):
        # 0 retries is valid (disables the _ascore retry loop; the openai
        # client's own max_retries still applies).
        
        cfg = RewardConfig(judge=_judge(), judge_max_retries=0)
        assert cfg.judge_max_retries == 0

    def test_judge_max_retries_negative_rejected(self):
        with pytest.raises(ValueError):
            RewardConfig(judge=_judge(), judge_max_retries=-1)

    def test_embedding_concurrency_defaults_to_judge(self):
        # When not specified, embedding_concurrency falls back to judge_concurrency.
        
        cfg = RewardConfig(judge=_judge(), judge_concurrency=6)
        assert cfg.embedding_concurrency == 6

    def test_embedding_concurrency_explicit(self):
        # Explicit embedding_concurrency is preserved independently.
        
        cfg = RewardConfig(
            judge=_judge(), judge_concurrency=8, embedding_concurrency=2,
        )
        assert cfg.embedding_concurrency == 2
        assert cfg.judge_concurrency == 8

    def test_embedding_concurrency_zero_rejected(self):
        with pytest.raises(ValueError):
            RewardConfig(judge=_judge(), embedding_concurrency=0)

    def test_describe_includes_new_fields(self):
        cfg = RewardConfig(
            judge=_judge(),
            judge_max_tokens=8192,
            judge_max_retries=3,
            known_tool_names=frozenset({"read", "write"}),
            compute_answer_correctness=False,
            embedding_concurrency=2,
        )
        d = cfg.describe()
        assert d["judge_max_tokens"] == 8192
        assert d["judge_max_retries"] == 3
        assert d["embedding_concurrency"] == 2
        assert d["known_tool_names_count"] == 2
        assert d["compute_answer_correctness"] is False

    def test_describe_judge_is_none_handled(self):
        # When judge is None, describe() must still work (logs a None judge).
        
        d = RewardConfig().describe()
        assert d["judge"] is None
        assert d["embeddings"]["local_embedding_model"] == "BAAI/bge-m3"


class TestLocalEmbeddingModel:
    def test_default_is_bge_m3(self):
        # The embedding model defaults to bge-m3 (no endpoint required).
        # bge-m3.
        cfg = RewardConfig()
        assert cfg.local_embedding_model == "BAAI/bge-m3"
        assert cfg.local_embedding_device is None  # auto-detect

    def test_custom_model(self):
        cfg = RewardConfig(local_embedding_model="intfloat/multilingual-e5-large")
        assert cfg.local_embedding_model == "intfloat/multilingual-e5-large"

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match="local_embedding_model"):
            RewardConfig(local_embedding_model="")

    def test_describe_with_local_model(self):
        cfg = RewardConfig(
            local_embedding_model="BAAI/bge-m3",
            local_embedding_device="cuda",
        )
        d = cfg.describe()
        assert d["embeddings"]["local_embedding_model"] == "BAAI/bge-m3"
        assert d["embeddings"]["local_embedding_device"] == "cuda"

    def test_describe_with_local_model_auto_device(self):
        cfg = RewardConfig(local_embedding_model="BAAI/bge-m3")
        d = cfg.describe()
        assert d["embeddings"]["local_embedding_device"] == "auto"


class TestFluencyAndCodeCheckConfig:
    def test_fluency_defaults(self):
        cfg = RewardConfig()
        assert cfg.local_fluency_model == "agentlans/snowflake-arctic-xs-grammar-classifier"
        assert cfg.local_fluency_device is None
        assert cfg.enable_fluency is True
        assert cfg.enable_code_check is True
        assert cfg.enable_nerel_f1 is True
        assert cfg.fluency_calibrator_path is None
        # Default allowlist includes the NL classes + coder (prose).
        assert "coder" in cfg.fluency_classes
        assert "menon_qa" in cfg.fluency_classes
        assert "nerel_instruct" in cfg.fluency_classes  # gated to prose sub-configs
        # Tool-use / code-only / short-answer classes are NOT in the default.
        assert "opencode_gen" not in cfg.fluency_classes
        assert "rummlu_pro" not in cfg.fluency_classes
        # factual_qa was dropped: its terse one-liners carry no prose signal
        # now that single sentences are scoreable.
        assert "factual_qa" not in cfg.fluency_classes

    def test_fluency_subtasks_default(self):
        cfg = RewardConfig()
        # nerel_instruct → only the prose sub-configs (definitions +
        # normalization); extraction sub-configs are scored by the F1 instead.
        assert set(cfg.fluency_subtasks["nerel_instruct"]) == {
            "entity_definition", "relation_definition", "entity_normalization",
        }

    def test_fluency_subtasks_dead_config_rejected(self):
        # A subtask key NOT in fluency_classes is dead config → rejected.
        
        with pytest.raises(ValueError, match="dead config"):
            RewardConfig(
                fluency_classes=frozenset({"menon_qa"}),
                fluency_subtasks={"coder": frozenset({"x"})},
            )

    def test_fluency_subtasks_unknown_class_rejected(self):
        with pytest.raises(ValueError, match="fluency_subtasks keys"):
            RewardConfig(fluency_subtasks={"not_a_class": frozenset({"x"})})

    def test_fluency_concurrency_defaults_to_embedding(self):
        # Like the embedder, the fluency model shares the policy GPU; its
        # concurrency falls back to embedding_concurrency when unset.
        
        cfg = RewardConfig(embedding_concurrency=4)
        assert cfg.fluency_concurrency == 4

    def test_fluency_concurrency_explicit(self):
        cfg = RewardConfig(embedding_concurrency=4, fluency_concurrency=2)
        assert cfg.fluency_concurrency == 2

    def test_fluency_concurrency_zero_rejected(self):
        with pytest.raises(ValueError, match="fluency_concurrency"):
            RewardConfig(fluency_concurrency=0)

    def test_enable_fluency_empty_model_rejected(self):
        with pytest.raises(ValueError, match="local_fluency_model"):
            RewardConfig(enable_fluency=True, local_fluency_model="")

    def test_disable_fluency_allows_empty_model(self):
        # When fluency is disabled, the model field is irrelevant.
        
        cfg = RewardConfig(enable_fluency=False, local_fluency_model="")
        assert cfg.enable_fluency is False

    def test_unknown_fluency_class_rejected(self):
        with pytest.raises(ValueError, match="Unknown task class"):
            RewardConfig(fluency_classes=frozenset({"menon_qa", "not_a_class"}))

    def test_describe_includes_fluency_block(self):
        cfg = RewardConfig(
            local_fluency_device="cuda",
            fluency_concurrency=2,
            enable_code_check=False,
        )
        d = cfg.describe()
        assert d["fluency"]["local_fluency_model"] == "agentlans/snowflake-arctic-xs-grammar-classifier"
        assert d["fluency"]["local_fluency_device"] == "cuda"
        assert d["fluency"]["enable_fluency"] is True
        assert d["fluency"]["enable_code_check"] is False
        assert d["fluency_concurrency"] == 2


class TestScoredMetricsShape:
    def test_minimal(self):
        # SemSim + combined only; AC and format_score default to None.
        
        m = ScoredMetrics(semantic_similarity=0.7, combined=0.7)
        assert m.answer_correctness is None
        assert m.format_score is None
        assert m.failed_metrics == ()

    def test_as_dict_roundtrip(self):
        m = ScoredMetrics(
            semantic_similarity=0.7,
            combined=0.6,
            answer_correctness=0.5,
            format_score=0.8,
            failed_metrics=("answer_correctness",),
        )
        d = m.as_dict()
        assert d["semantic_similarity"] == 0.7
        assert d["answer_correctness"] == 0.5
        assert d["format_score"] == 0.8
        assert d["combined"] == 0.6
        assert d["failed_metrics"] == ["answer_correctness"]

    def test_as_dict_keys(self):
        # All twelve metric fields + failed_metrics are present.
        
        m = ScoredMetrics(semantic_similarity=0.7, combined=0.7)
        assert set(m.as_dict()) == {
            "semantic_similarity", "answer_correctness",
            "rubric_score", "format_score", "fluency", "code_check",
            "nerel_f1", "docsem_score", "rummlu_em", "schema_check",
            "rerank_em", "combined", "failed_metrics",
        }

    def test_defaults_fluency_and_code_check_none(self):
        # New signals default to None (excluded from the harmonic combine).
        
        m = ScoredMetrics(semantic_similarity=0.7, combined=0.7)
        assert m.fluency is None
        assert m.code_check is None
        assert m.nerel_f1 is None
        assert m.docsem_score is None
        assert m.rummlu_em is None
        assert m.schema_check is None
        assert m.rerank_em is None

    def test_format_epsilon_floor_validation(self):
        # 0.0 = off (default, strict semantics); (0, 1) — valid; 1.0+ and
        # negatives are rejected.
        
        assert RewardConfig().format_epsilon_floor == 0.0
        assert RewardConfig(format_epsilon_floor=0.15).format_epsilon_floor == 0.15
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError, match="format_epsilon_floor"):
                RewardConfig(format_epsilon_floor=bad)

    def test_schema_check_defaults_off_rummlu_em_on(self):
        # schema_check is OFF by default (the anchors never set it; only the
        # GRPO in-loop config enables it); rummlu_em is ON by default (the
        # NEREL-F1/docsem replacement pattern, never a hold-out class).
        # schema_check
        # rummlu_em
        cfg = RewardConfig()
        assert cfg.enable_schema_check is False
        assert cfg.enable_rummlu_em is True
        assert cfg.enable_rerank_em is True
        assert cfg.known_tool_schemas == {}

    def test_meno_final_in_fluency_allowlist(self):
        # The Meno-stand final answers are RU prose (a menon_qa kin); the
        # rewrite/rerank classes stay out.
        
        cfg = RewardConfig()
        assert "meno_final" in cfg.fluency_classes
        assert "meno_rewrite" not in cfg.fluency_classes
        assert "meno_rerank" not in cfg.fluency_classes

    def test_as_dict_empty_when_no_failures(self):
        m = ScoredMetrics(semantic_similarity=0.7, combined=0.7)
        assert m.as_dict()["failed_metrics"] == []

    def test_semsim_optional_for_nerel_extraction(self):
        # For NEREL extraction sub-configs SemSim is None (F1 replaces it).
        
        m = ScoredMetrics(semantic_similarity=None, combined=0.8, nerel_f1=0.8)
        assert m.semantic_similarity is None
        assert m.nerel_f1 == 0.8
