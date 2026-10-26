'Deterministic hybrid retrieval index (BM25 + dense) for the agent suite.\n\n\nStep 1 of ``specifications/2026-10-17-agent-suite-glm53-validation.md``: the\nfrozen ``searchdocs`` tool backend shared by the GLM-5.3 teacher pass and the\npolicy anchors (ONE snapshot, so the retrieval environment is identical by\nconstruction), and later the retrieval tool of the training-side tool classes.\n\n\nDesign invariants\n\n1. **BM25 (Okapi).** Non-negative idf ``ln((N - df + 0.5)/(df + 0.5) + 1)``;\n   query terms are unique (the standard Okapi sum).\n   **BM25 (Okapi).**\n2. **Tokenization.** Lowercase → regex ``\\w+`` tokens → per-token Snowball\n   stemming routed by script: Cyrillic → Russian, ASCII-alpha → English,\n   other (digits etc.) → verbatim lowercase. Both Snowball stemmers are\n   rule-based and need NO ``nltk.download`` data (the ``nerel_f1`` precedent);\n   lazy imports keep the module importable without the ``train`` extra.\n  .rule-based, ``nltk.download``\n3. **Dense side.** The index stores L2-normalized document vectors produced by\n   a PLUGGABLE encoder callable; the module has zero heavy imports. At runtime\n   the caller passes a wrapper over the reward-path bge-m3 singleton\n   (spec section 5: no second model in VRAM); offline tests inject a trivial\n   deterministic encoder. Queries are embedded at search time through the same\n   callable.\n4. **Fusion.** Weighted Reciprocal Rank Fusion:\n   ``score = Σ w_i / (rrf_k + rank_i)`` over the top ``candidate_pool`` of each\n   channel. Ranks are computed over ``(-score, document order)``, so identical\n   inputs give identical rankings — the determinism gate of the spec.\n5. **Snapshot.** ``snapshot.npz`` (float32 vectors, int32/int64 postings) +\n   ``snapshot.json`` (config, doc_ids, terms, avgdl) inside one directory.\n   :meth:`HybridIndex.save` returns CONTENT digests (sha256 over canonical\n   logical bytes, NOT file bytes — zip containers carry timestamps), which the\n   suite manifest pins. Loading requires the encoder callable again: queries\n   are always embedded at search time.'

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Encoder contract: a list of texts → an (N, D) array-like of float32 rows.

EncodeFn = Callable[[list[str]], Any]

#: Snapshot format version (bump on incompatible layout changes).

FORMAT_VERSION: int = 1

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_stemmers: dict[str, Any] = {}


def _snowball(language: str) -> Any:
    'Lazily build + cache a rule-based Snowball stemmer (no nltk data).\n\n    nltk).'
    stemmer = _stemmers.get(language)
    if stemmer is None:
        from nltk.stem.snowball import SnowballStemmer  # no downloads needed.

        stemmer = SnowballStemmer(language)
        _stemmers[language] = stemmer
    return stemmer


def stem_token(token: str) -> str:
    'Route one raw token by script and stem it deterministically.'
    lowered = token.lower()
    if any("а" <= ch <= "я" or ch == "ё" for ch in lowered):
        return _snowball("russian").stem(lowered)
    if lowered.isascii() and lowered.isalpha():
        return _snowball("english").stem(lowered)
    return lowered


def tokenize_stems(text: str) -> list[str]:
    'Lowercase word tokens → per-token stems (the whole indexing lexicon).'
    return [stem_token(token) for token in _TOKEN_RE.findall(text)]


def _unit_rows(matrix: Any, *, what: str) -> np.ndarray:
    'Validate an (N, D) encoder output and L2-normalize its rows.'
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{what}: encoder must return a 2-D array, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _term_ids_array(n_terms: int, offsets: list[int]) -> np.ndarray:
    'Term index per posting entry, aligned with the postings docs array.'
    ids: list[int] = []
    for term_pos in range(n_terms):
        ids.extend([term_pos] * (offsets[term_pos + 1] - offsets[term_pos]))
    return np.asarray(ids, dtype=np.int32)


@dataclass(frozen=True)
class RetrievalConfig:
    'Fusion and BM25 parameters; pinned into the snapshot JSON.'

    k1: float = 1.5
    b: float = 0.75
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    #: Per-channel candidate pool for RRF (auto-raised to ``top_k``).
    
    candidate_pool: int = 100


@dataclass(frozen=True)
class RetrievalHit:
    'One fused search result (diagnostic ranks included).'

    doc_id: str
    doc_index: int
    # : Weighted RRF score; larger is better
    score: float
    #: 1-based channel ranks; ``None`` when the channel missed the pool.
    
    bm25_rank: int | None = None
    dense_rank: int | None = None


class OpenAIEmbeddingsEncoder:
    'Encoder over an OpenAI-compatible ``/v1/embeddings`` endpoint.\n\n\n    The served-embeddings mode of the agent-suite index (spec section 16):\n    the batched HTTP client cuts ``/v1/embeddings`` requests itself, so\n    :meth:`HybridIndex.build` receives ONE callable regardless of the\n    corpus size (326k chunks). Returns RAW float32 vectors — normalization\n    stays in ``_unit_rows`` (the :class:`HybridIndex` contract). The\n    ``openai`` import is lazy (train extra); tests inject a stub ``client``.\n\n    :meth:`HybridIndex.build`\n    ``client``.'

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "EMPTY",
        batch_size: int = 64,
        timeout_s: float = 300.0,
        max_retries: int = 2,
        client: Any = None,
    ) -> None:
        self.base_url = base_url
        self.model_id = model_id
        self.api_key = api_key
        self.batch_size = max(1, int(batch_size))
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # lazy: train-extra dependency.

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout_s,
                max_retries=self.max_retries,
            )
        return self._client

    def __call__(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [str(text) for text in texts[start : start + self.batch_size]]
            response = client.embeddings.create(model=self.model_id, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise ValueError(
                    f'embeddings endpoint returned {len(ordered)} vectors for {len(batch)}'
                )
            rows.extend(
                np.asarray(item.embedding, dtype=np.float32) for item in ordered
            )
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(rows)


class HybridIndex:
    'BM25 + dense hybrid index over a frozen list of documents.'

    def __init__(
        self,
        *,
        config: RetrievalConfig,
        encode: EncodeFn,
        doc_ids: list[str],
        doc_lengths: np.ndarray,
        avgdl: float,
        vectors: np.ndarray,
        terms: list[str],
        postings_term: np.ndarray,
        postings_doc: np.ndarray,
        postings_tf: np.ndarray,
        term_offsets: np.ndarray,
    ) -> None:
        n_docs = len(doc_ids)
        if vectors.shape[0] != n_docs or doc_lengths.shape[0] != n_docs:
            raise ValueError("doc_ids, doc_lengths and vectors must agree on the document count")
        if postings_doc.shape != postings_tf.shape or postings_term.shape != postings_doc.shape:
            raise ValueError("postings arrays must share one length")
        if len(term_offsets) != len(terms) + 1:
            raise ValueError("term_offsets must have len(terms) + 1 entries")
        self.config = config
        self._encode = encode
        self.doc_ids: tuple[str, ...] = tuple(doc_ids)
        self._doc_lengths = doc_lengths
        self._avgdl = float(avgdl)
        self._vectors = vectors
        self._terms: tuple[str, ...] = tuple(terms)
        self._term_index: dict[str, int] = {term: i for i, term in enumerate(terms)}
        self._postings_term = postings_term
        self._postings_doc = postings_doc
        self._postings_tf = postings_tf
        self._term_offsets = term_offsets

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        texts: list[str],
        doc_ids: list[str],
        encode: EncodeFn,
        config: RetrievalConfig | None = None,
        vectors: np.ndarray | None = None,
    ) -> HybridIndex:
        'Index ``texts`` in the given order (doc order fixes all tie-breaks).\n\n        the ``encode`` call — the batched HTTP client cuts requests itself\n        (the agent-suite index over 326k chunks);'
        texts = list(texts)
        doc_ids = list(doc_ids)
        if not texts:
            raise ValueError("cannot build an index over an empty document list")
        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must have the same length")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids must be unique")
        config = config or RetrievalConfig()

        stemmed = [tokenize_stems(text) for text in texts]
        doc_lengths = np.asarray([len(stems) for stems in stemmed], dtype=np.int32)
        total = int(doc_lengths.sum())
        avgdl = total / len(stemmed)

        per_term: dict[str, list[tuple[int, int]]] = {}
        for doc_index, stems in enumerate(stemmed):
            for term, tf in Counter(stems).items():
                per_term.setdefault(term, []).append((doc_index, tf))
        terms = sorted(per_term)
        docs_chunks: list[int] = []
        tf_chunks: list[int] = []
        offsets: list[int] = [0]
        for term in terms:
            for doc_index, tf in per_term[term]:
                docs_chunks.append(doc_index)
                tf_chunks.append(tf)
            offsets.append(len(docs_chunks))

        if vectors is None:
            vectors = _unit_rows(encode(texts), what="build")
        else:
            vectors = _unit_rows(vectors, what="vectors")
        return cls(
            config=config,
            encode=encode,
            doc_ids=doc_ids,
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            vectors=vectors,
            terms=terms,
            postings_term=_term_ids_array(len(terms), offsets),
            postings_doc=np.asarray(docs_chunks, dtype=np.int32),
            postings_tf=np.asarray(tf_chunks, dtype=np.int32),
            term_offsets=np.asarray(offsets, dtype=np.int64),
        )

    # ----------------------------------------------------------------- search

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        'Fuse BM25 and dense rankings; ties break by document order.'
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        if not query.strip():
            return []
        n_docs = len(self.doc_ids)
        pool = max(self.config.candidate_pool, top_k)

        bm25_scores = self._bm25_scores(query)
        bm25_top = sorted(bm25_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:pool]
        sims = self._vectors @ self._encode_query(query)
        dense_order = np.lexsort((np.arange(n_docs), -sims))[:pool]

        fused: dict[int, float] = {}
        bm25_ranks: dict[int, int] = {}
        dense_ranks: dict[int, int] = {}
        for rank, (doc_index, _) in enumerate(bm25_top, start=1):
            bm25_ranks[doc_index] = rank
            fused[doc_index] = fused.get(doc_index, 0.0) + self.config.bm25_weight / (self.config.rrf_k + rank)
        for rank, doc_index in enumerate(dense_order.tolist(), start=1):
            dense_ranks[doc_index] = rank
            fused[doc_index] = fused.get(doc_index, 0.0) + self.config.dense_weight / (self.config.rrf_k + rank)
        top = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [
            RetrievalHit(
                doc_id=self.doc_ids[doc_index],
                doc_index=doc_index,
                score=score,
                bm25_rank=bm25_ranks.get(doc_index),
                dense_rank=dense_ranks.get(doc_index),
            )
            for doc_index, score in top
        ]

    def _bm25_scores(self, query: str) -> dict[int, float]:
        "Okapi BM25 accumulation over the query's unique stems."
        query_counts = Counter(tokenize_stems(query))
        if not query_counts:
            return {}
        n_docs = len(self.doc_ids)
        avgdl = self._avgdl if self._avgdl > 0.0 else 1.0
        scores: dict[int, float] = {}
        k1, b = self.config.k1, self.config.b
        for term in query_counts:
            pos = self._term_index.get(term)
            if pos is None:
                continue
            start, end = int(self._term_offsets[pos]), int(self._term_offsets[pos + 1])
            df = end - start
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            docs = self._postings_doc[start:end].tolist()
            tfs = self._postings_tf[start:end].tolist()
            for doc_index, tf in zip(docs, tfs, strict=True):
                norm = tf + k1 * (1.0 - b + b * float(self._doc_lengths[doc_index]) / avgdl)
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * tf * (k1 + 1.0) / norm
        return scores

    def _encode_query(self, query: str) -> np.ndarray:
        'Embed one query through the shared encoder and normalize it.'
        return _unit_rows(self._encode([query]), what="query")[0]

    # --------------------------------------------------------------- snapshot

    def save(self, directory: str | Path) -> dict[str, Any]:
        'Write ``snapshot.npz`` + ``snapshot.json``; return content digests.'
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        npz_path = out / "snapshot.npz"
        json_path = out / "snapshot.json"
        np.savez(
            npz_path,
            vectors=self._vectors,
            doc_lengths=self._doc_lengths,
            postings_term=self._postings_term,
            postings_doc=self._postings_doc,
            postings_tf=self._postings_tf,
            term_offsets=self._term_offsets,
        )
        payload = {
            "format_version": FORMAT_VERSION,
            "config": asdict(self.config),
            "doc_ids": list(self.doc_ids),
            "terms": list(self._terms),
            "avgdl": self._avgdl,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
        )
        return {
            "directory": str(out),
            "npz": str(npz_path),
            "json": str(json_path),
            "digest": self.content_digest(),
        }

    def content_digest(self) -> str:
        'sha256 over canonical logical bytes (zip-timestamp-free).'
        hasher = hashlib.sha256()
        hasher.update(self._vectors.tobytes())
        hasher.update(self._doc_lengths.tobytes())
        hasher.update(self._postings_term.tobytes())
        hasher.update(self._postings_doc.tobytes())
        hasher.update(self._postings_tf.tobytes())
        hasher.update(self._term_offsets.tobytes())
        hasher.update(json.dumps(self.config_args(), sort_keys=True).encode("utf-8"))
        hasher.update(json.dumps(list(self.doc_ids), ensure_ascii=False).encode("utf-8"))
        hasher.update(json.dumps(list(self._terms), ensure_ascii=False).encode("utf-8"))
        hasher.update(repr(self._avgdl).encode("utf-8"))
        return f"sha256:{hasher.hexdigest()}"

    def config_args(self) -> dict[str, Any]:
        'The config as a plain dict (snapshot/manifest payload).'
        return asdict(self.config)

    @classmethod
    def load(cls, directory: str | Path, encode: EncodeFn) -> HybridIndex:
        'Rebuild an index from a snapshot; queries embed via ``encode``.\n\n        ``encode``.'
        directory = Path(directory)
        payload = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"snapshot format_version {payload.get('format_version')!r} != {FORMAT_VERSION}"
            )
        with np.load(directory / "snapshot.npz") as npz:
            return cls(
                config=RetrievalConfig(**payload["config"]),
                encode=encode,
                doc_ids=list(payload["doc_ids"]),
                doc_lengths=npz["doc_lengths"],
                avgdl=float(payload["avgdl"]),
                vectors=npz["vectors"],
                terms=list(payload["terms"]),
                postings_term=npz["postings_term"],
                postings_doc=npz["postings_doc"],
                postings_tf=npz["postings_tf"],
                term_offsets=npz["term_offsets"],
            )
