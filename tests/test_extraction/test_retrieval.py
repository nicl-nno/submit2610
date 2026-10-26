'Offline tests of the hybrid retrieval index (no network, no GPU).\n\n\nCovers the step-1 gates of\n``specifications/2026-10-17-agent-suite-glm53-validation.md``: determinism of\nbuild/search/snapshot digests, save/load round-trip, script-routed stemming,\nRRF fusion diagnostics and argument validation.'

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from llm_training.extraction.retrieval import (
    HybridIndex,
    RetrievalConfig,
    stem_token,
    tokenize_stems,
)

_RU_DOCS = [
    "В центре города открыли новый мост через реку.",  
    "Река протекает через горные ущелья на юге.",  
    "Отчёт компании за год: рост выручки и новые рынки.",  
    "Мост через реку соединил два берега города.",  
]


def _hash_encoder(dim: int = 64):
    'Deterministic hashed bag-of-words encoder (offline bge-m3 stand-in).'

    def encode(texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                number = int.from_bytes(digest, "big")
                rows[i, number % dim] += 1.0 + float((number >> 11) % 7)
        return rows

    return encode


@pytest.fixture()
def index() -> HybridIndex:
    ids = [f"doc-{i}" for i in range(len(_RU_DOCS))]
    return HybridIndex.build(_RU_DOCS, ids, encode=_hash_encoder())


# --------------------------------------------------------------- tokenization


def test_stemming_routes_scripts() -> None:
    assert stem_token("Города") == stem_token("город")
    assert stem_token("компаний") == stem_token("компания")
    assert stem_token("cities") == stem_token("city")
    assert stem_token("2024") == "2024"


def test_tokenize_stems_full_pipeline() -> None:
    assert tokenize_stems("Города, реки. 2024!") == tokenize_stems("города реки 2024")


# ------------------------------------------------------------------- ranking


def test_search_ranks_relevant_doc_first(index: HybridIndex) -> None:
    hits = index.search("новый мост в городе", top_k=2)
    assert hits[0].doc_id == "doc-0"
    assert hits[0].bm25_rank == 1
    assert hits[0].dense_rank is not None


def test_stemmed_inflection_matches(index: HybridIndex) -> None:
    
    
    hits = index.search("города", top_k=4)
    assert {hits[0].doc_id, hits[1].doc_id} == {"doc-0", "doc-3"}
    for hit in hits:
        if hit.doc_id in {"doc-0", "doc-3"}:
            assert hit.bm25_rank is not None


def test_empty_query_returns_empty(index: HybridIndex) -> None:
    assert index.search("", top_k=3) == []
    assert index.search("   ", top_k=3) == []


def test_top_k_validation_and_clamp(index: HybridIndex) -> None:
    with pytest.raises(ValueError):
        index.search("мост", top_k=0)
    hits = index.search("мост", top_k=99)
    assert len(hits) == len(_RU_DOCS)


def test_tie_break_is_document_order() -> None:
    texts = ["одинаковый текст"] * 3
    ids = ["b", "a", "c"]
    built = HybridIndex.build(texts, ids, encode=_hash_encoder())
    hits = built.search("одинаковый", top_k=3)
    assert [hit.doc_index for hit in hits] == [0, 1, 2]
    assert [hit.doc_id for hit in hits] == ["b", "a", "c"]


def test_duplicate_doc_ids_rejected() -> None:
    with pytest.raises(ValueError):
        HybridIndex.build(["a", "b"], ["x", "x"], encode=_hash_encoder())


# ------------------------------------------------------------------- fusion


def test_hit_carries_both_channel_ranks(index: HybridIndex) -> None:
    hit = index.search("отчёт компании", top_k=1)[0]
    assert hit.doc_id == "doc-2"
    assert hit.bm25_rank == 1 and hit.dense_rank is not None
    max_score = (
        RetrievalConfig().bm25_weight + RetrievalConfig().dense_weight
    ) / RetrievalConfig().rrf_k
    assert 0.0 < hit.score <= max_score


def test_candidate_pool_limits_fusion() -> None:
    config = RetrievalConfig(candidate_pool=1)
    texts = ["тема альфа", "тема бета", "тема гамма"]
    ids = [str(i) for i in range(len(texts))]
    built = HybridIndex.build(texts, ids, encode=_hash_encoder(), config=config)
    hits = built.search("тема", top_k=3)
    
    
    assert len(hits) == 3
    assert hits[0].doc_id == "0"


# ------------------------------------------------------------------ snapshot


def test_save_load_roundtrip(tmp_path) -> None:
    encode = _hash_encoder()
    built = HybridIndex.build(_RU_DOCS, [f"d{i}" for i in range(len(_RU_DOCS))], encode=encode)
    before = built.search("мост через реку", top_k=4)
    saved = built.save(tmp_path / "snap")
    loaded = HybridIndex.load(tmp_path / "snap", encode=encode)
    after = loaded.search("мост через реку", top_k=4)
    assert before == after
    assert saved["digest"].startswith("sha256:")
    assert (tmp_path / "snap" / "snapshot.npz").exists()
    assert (tmp_path / "snap" / "snapshot.json").exists()


def test_snapshot_digest_is_stable(tmp_path) -> None:
    encode = _hash_encoder()
    ids = [f"d{i}" for i in range(len(_RU_DOCS))]
    first = HybridIndex.build(_RU_DOCS, ids, encode=encode)
    second = HybridIndex.build(_RU_DOCS, ids, encode=_hash_encoder())
    saved_first = first.save(tmp_path / "one")
    saved_second = second.save(tmp_path / "two")
    assert saved_first["digest"] == saved_second["digest"]
    assert first.content_digest() == second.content_digest()


def test_snapshot_records_config(tmp_path) -> None:
    import json

    config = RetrievalConfig(rrf_k=17, candidate_pool=5)
    built = HybridIndex.build(_RU_DOCS, [str(i) for i in range(len(_RU_DOCS))], encode=_hash_encoder(), config=config)
    built.save(tmp_path / "snap")
    payload = json.loads((tmp_path / "snap" / "snapshot.json").read_text(encoding="utf-8"))
    assert payload["config"]["rrf_k"] == 17
    assert payload["config"]["candidate_pool"] == 5
    loaded = HybridIndex.load(tmp_path / "snap", encode=_hash_encoder())
    assert loaded.config == config


def test_build_with_precomputed_vectors_matches_encode(tmp_path) -> None:
    encode = _hash_encoder()
    ids = [f"d{i}" for i in range(len(_RU_DOCS))]
    manual = np.asarray(encode(_RU_DOCS), dtype=np.float32)
    via_encode = HybridIndex.build(_RU_DOCS, ids, encode=encode)
    via_vectors = HybridIndex.build(_RU_DOCS, ids, encode=encode, vectors=manual)
    for query in ("мост через реку", "отчёт компании", "горные ущелья"):
        assert via_encode.search(query, top_k=4) == via_vectors.search(query, top_k=4)


def test_openai_embeddings_encoder_batches_and_aligns() -> None:
    import types

    class _StubEmbeddings:
        def __init__(self, dim: int = 8) -> None:
            self.dim = dim
            self.requests: list[list[str]] = []

        def create(self, model: str, input):
            self.requests.append(list(input))
            items = [
                types.SimpleNamespace(
                    index=i, embedding=[float(len(text) % 5 + 1)] * self.dim
                )
                for i, text in enumerate(input)
            ]
            return types.SimpleNamespace(data=items)

    from llm_training.extraction.retrieval import OpenAIEmbeddingsEncoder

    stub = _StubEmbeddings()
    encoder = OpenAIEmbeddingsEncoder(
        base_url="http://localhost:8025/v1",
        model_id="bge-m3",
        batch_size=2,
        client=types.SimpleNamespace(
            embeddings=types.SimpleNamespace(create=stub.create)
        ),
    )
    vectors = encoder(["a", "bb", "ccc"])
    assert vectors.shape == (3, 8)
    assert len(stub.requests) == 2
    assert stub.requests[0] == ["a", "bb"]
    assert stub.requests[1] == ["ccc"]
    assert float(np.linalg.norm(vectors[0])) > 0.0
