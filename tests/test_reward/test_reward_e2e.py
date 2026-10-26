'End-to-end reward test against a REAL judge server + in-process embeddings.\n\n\nThis test is marked ``@pytest.mark.gpu`` and is deselected by default\n(``pytest -m "not gpu"``). It exercises the full RAGAS-backed scorer end to end:\n\n* **Semantic Similarity** — always (embeddings via ``sentence-transformers`` in\n  process; the model is loaded onto the GPU/CPU — this is why the test carries\n  the ``gpu`` marker; it never needs an embedding HTTP server);\n* **Answer Correctness** — if the judge endpoint (``JUDGE_*``) is provided;\n* **format_check** — on the tool-use path (``test_tool_use_path_produces_format_score``);\n* **fluency** — on a natural-language class (``test_fluency_path_produces_score``);\n  the in-process grammar classifier is loaded here too (a tiny BERT, ~22M\n  params), so the ``gpu`` marker covers BOTH in-process models.\n\n\nRun it manually against your served judge, e.g.::\n\n    JUDGE_BASE_URL=http://localhost:8001/v1 JUDGE_MODEL=qwen2.5-32b-instruct \\\n    LOCAL_EMBEDDING_DEVICE=cuda LOCAL_FLUENCY_DEVICE=cuda \\\n    pytest -m gpu tests/test_reward/test_reward_e2e.py\n\nIf ``JUDGE_*`` is missing the test runs in embedding-only mode (AC stays None);\nthe fluency test still runs (it does not need the judge).'

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gpu]

_JUDGE_ENV = ("JUDGE_BASE_URL", "JUDGE_MODEL")


def _have_env(env: tuple[str, ...]) -> bool:
    return all(os.environ.get(v) for v in env)


@pytest.fixture(scope="module")
def reward_cfg():
    from llm_training.reward.config import RewardConfig
    from llm_training.runtime.endpoints import OpenAIEndpoint

    # Embeddings AND the fluency classifier are always in-process
    # (sentence-transformers + transformers on the chosen device). The judge is
    # an external HTTP endpoint.
    
    if _have_env(_JUDGE_ENV):
        judge_ep = OpenAIEndpoint(
            base_url=os.environ["JUDGE_BASE_URL"],
            model=os.environ["JUDGE_MODEL"],
            api_key=os.environ.get("JUDGE_API_KEY"),
            kind="judge",
        )
    else:
        judge_ep = None
    return RewardConfig(
        local_embedding_model=os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3"),
        local_embedding_device=os.environ.get("LOCAL_EMBEDDING_DEVICE"),
        local_fluency_model=os.environ.get(
            "LOCAL_FLUENCY_MODEL", "agentlans/snowflake-arctic-xs-grammar-classifier"
        ),
        local_fluency_device=os.environ.get("LOCAL_FLUENCY_DEVICE"),
        judge=judge_ep,
        compute_answer_correctness=judge_ep is not None,
        judge_concurrency=4,
    )


def test_semantic_similarity_on_plain_example(reward_cfg):
    # A plain example of a non-tool, non-fluency class (rummlu_pro is in
    # neither TOOL_USE_CLASSES nor the fluency allowlist): SemSim always
    # computed; AC computed iff judge is set; format/code/fluency stay None.
    
    # TOOL_USE_CLASSES
    
    from llm_training.reward.judge import make_ragas_scorer
    from llm_training.runtime.reward_service import RewardService, ScoreRequest

    scorer = make_ragas_scorer(reward_cfg)
    with RewardService(scorer) as svc:
        results = svc.score_batch(
            [
                ScoreRequest(
                    prompt=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is 2 + 2?"},
                    ],
                    completion="4",
                    task_class="rummlu_pro",
                    reference="4",
                    extra={},
                )
            ]
        )
    m = results[0]
    assert 0.0 <= m.semantic_similarity <= 1.0
    assert 0.0 <= m.combined <= 1.0
    assert m.format_score is None  # non-tool class
    # AC is populated only when a judge endpoint was configured.
    
    if reward_cfg.judge is not None:
        assert 0.0 <= m.answer_correctness <= 1.0
    else:
        assert m.answer_correctness is None


def test_tool_use_path_produces_format_score(reward_cfg):
    # A tool-use example must produce a non-None format_score.
    # tool-use
    from llm_training.reward.judge import make_ragas_scorer
    from llm_training.runtime.reward_service import RewardService, ScoreRequest

    response = '<tool_call>{"name": "read", "arguments": {"path": "x"}}</tool_call>'
    scorer = make_ragas_scorer(reward_cfg)
    with RewardService(scorer) as svc:
        results = svc.score_batch(
            [
                ScoreRequest(
                    prompt=[
                        {"role": "system", "content": "You can call tools."},
                        {"role": "user", "content": "Read the file."},
                    ],
                    completion=response,
                    task_class="opencode_gen",
                    reference=response,
                    extra={},
                )
            ]
        )
    m = results[0]
    assert 0.0 <= m.combined <= 1.0
    assert m.format_score is not None  # tool-use class


def test_fluency_path_produces_score(reward_cfg):
    # A natural-language answer (menon_qa is in the fluency allowlist) whose
    # sentences each have ≥ MIN_FLUENCY_WORDS words triggers the in-process
    # grammar classifier. This is the only @gpu test that actually loads the
    # fluency model. It does NOT need the judge (fluency is an in-process
    # signal, independent of AC).
    
    # MIN_FLUENCY_WORDS
    
    from llm_training.reward.judge import make_ragas_scorer
    from llm_training.runtime.reward_service import RewardService, ScoreRequest

    scorer = make_ragas_scorer(reward_cfg)
    with RewardService(scorer) as svc:
        results = svc.score_batch(
            [
                ScoreRequest(
                    prompt=[
                        {"role": "system", "content": "Вы — полезный ассистент."},
                        {"role": "user", "content": "Расскажите о создании AU."},
                    ],
                    completion=(
                        "Anonymous University был основан в 1959 году. "
                        "Он быстро стал одним из ведущих вузов страны в области точных наук."
                    ),
                    task_class="menon_qa",
                    reference="AU основан в 1959 году.",
                    extra={},
                )
            ]
        )
    m = results[0]
    # Fluency must have been computed (not None) and lie in [0, 1].
    
    assert m.fluency is not None
    assert 0.0 <= m.fluency <= 1.0
    assert 0.0 <= m.combined <= 1.0
