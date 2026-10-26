'Collect the open sources of the GRPO-extraction corpus (offline, CPU).\n\n\nOne subcommand per source; each writes JSONL files the corpus builder\n(``scripts/build_extraction_corpus.py``) consumes. Pure helpers are\nnetwork-free and unit-tested; the ``datasets``/GitHub fetches live only\ninside the per-source subcommands.\n\nSources & licenses (the AGENTS.md license table is authoritative —\na source without a clear license is NOT taken)\n\n============  ============================================  ====================\nsubcommand    source                                        license\n============  ============================================  ====================\nwikipedia     HF ``wikimedia/wikipedia`` snapshots          CC BY-SA 4.0\ngutenberg     local ``.txt`` files from gutenberg.org       public domain\ngutenberg-dl  gutenberg.org ``.txt`` via Gutendex listing   public domain\nted           local transcripts (operator-supplied)         CC BY-NC-ND (table)\nfineweb2      HF ``epfml/FineWeb2-HQ-QA`` ``rus_Cyrl``      ODC-By v1.0\nnerel-bio     github ``nerel-ds/NEREL-BIO`` ``nerel-bio-    research use, cite\n              v1.0`` (brat annotations)                     Bioinformatics 2023\n============  ============================================  ====================\n\nRecord formats\n\n* chunk record (every subcommand; the builder\'s ``--input-jsonl`` format)::\n      {"text": ..., "style": ..., "lang": "ru"|"en", "source": ...,\n       "doc_id": ..., "ontology": null|"nerel_bio", "code_switched": bool}\n\n  ``code_switched`` marks the RU-EN mixed register (IT/engineering prose,\n  Latin-word share inside the band); the quota is per-subcommand\n  (``--code-switch-share``), the ``mix`` subcommand reports the totals.\n  ``code_switched``\n\n* QA record (``fineweb2 --qa-out``; shaped into ``fineweb_qa`` rows by the\n  dataset assembly)::\n      {"text": ..., "question": ..., "answer": ..., "source": ...,\n       "url": ..., "doc_id": ..., "lang": "ru"}\n\n* gold record (``nerel-bio --gold-out`` over dev/test; the frozen anchor /\n  offline-eval material — expert annotations in the list format)::\n      {"doc_id": ..., "text": ..., "ontology": "nerel_bio",\n       "gold": {"entities": [...], "relations": [...]}}\n\nEverything is deterministic under ``--seed`` (sorted inputs, seeded\nsampling, fixed shard order).\n\nUsage\n\n    python scripts/collect_extraction_corpus.py wikipedia         --out chunks_wiki.jsonl --max-docs 40000 --seed 42\n\n    python scripts/collect_extraction_corpus.py gutenberg-download         --out chunks_gut.jsonl --langs ru,en --max-docs 15000 --seed 42\n\n    python scripts/collect_extraction_corpus.py fineweb2         --out chunks_fw2.jsonl --qa-out qa_fw2.jsonl         --scan-rows 400000 --max-docs 30000 --seed 42\n\n    python scripts/collect_extraction_corpus.py nerel-bio         --out chunks_bio.jsonl --gold-out gold_bio.jsonl --ref master\n\n    python scripts/collect_extraction_corpus.py mix         --input chunks_wiki.jsonl --input chunks_fw2.jsonl         --input chunks_gut.jsonl --input chunks_bio.jsonl         --out chunks.jsonl --lang-share ru=0.7,en=0.3 --seed 42'

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import http.client
import json
import random
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Pure helpers (no network; unit-tested)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_LINE_RE = re.compile(r"- Question:\s*(.+?)\s+Answer:\s*(.+)$")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_RE = re.compile(r"[A-Za-z]")
_GUT_START_RE = re.compile(
    r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE
)
_GUT_END_RE = re.compile(
    r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE
)

#: The NEREL-BIO tarball on GitHub (codeload) — pinned ``--ref``.
_BIO_TARBALL_URL = "https://codeload.github.com/nerel-ds/NEREL-BIO/tar.gz/{ref}"
_BIO_SUBDIR = "nerel-bio-v1.0"

#: The Gutendex public-domain metadata API (paginated book listing, JSON).
_GUTENDEX_URL = "https://gutendex.com/books/"
#: The Gutenberg plain-text cache path (the FALLBACK — the naming varies by
#: upload age, e.g. ``<id>-0.txt`` for newer books; the Gutendex-provided
#: URL redirects to the actual file and is tried FIRST).
_GUTEN_CACHE_TPL = (
    "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
)

#: The User-Agent for the direct HTTP fetches (Gutendex and gutenberg.org
#: return 403 to the default python-urllib agent).
_USER_AGENT = "llm-training-corpus-collector/1.0"


def count_words(text: str) -> int:
    'Language-aware word count (Cyrillic + Latin).'
    return len(_WORD_RE.findall(text))


def detect_lang(text: str) -> str:
    '``ru`` when Cyrillic dominates, else ``en``'
    cyr = len(_CYR_RE.findall(text))
    lat = len(_LAT_RE.findall(text))
    return "ru" if cyr > lat else "en"


def cyrillic_share(text: str) -> float:
    'Share of Cyrillic letters among all letters'
    cyr = len(_CYR_RE.findall(text))
    lat = len(_LAT_RE.findall(text))
    total = cyr + lat
    return (cyr / total) if total else 0.0


def latin_word_share(text: str) -> float:
    'Share of words carrying at least one Latin letter\n\n    A word counts when it contains a Latin letter (``API``, ``\n    ``Kubernetes``); digit-only words and Cyrillic words do not. The signal\n    of RU-EN code switching (IT/engineering prose) — English prose sits at\n    ≈ 1.0, clean Russian near 0.0.'
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    latin = sum(1 for w in words if _LAT_RE.search(w))
    return latin / len(words)


def is_code_switched(
    text: str, min_share: float = 0.03, max_share: float = 0.5
) -> bool:
    'RU-EN code-switching band check (the IT/engineering register).\n\n    the text is clean Russian; above ``max_share`` it is plain English (or a\n    Latin-language document), not a mixed register.'
    share = latin_word_share(text)
    return min_share <= share <= max_share


def gutendex_candidates(payload: dict[str, Any]) -> list[tuple[int, str, str]]:
    'Extract ``(book_id, title, text_url)`` candidates from a Gutendex page.\n\n    Gutendex. Only ``media_type == "Text"`` entries carrying a\n    ``text/plain`` format qualify; ``charset=utf-8`` wins over other\n    charsets; the page order is preserved.'
    out: list[tuple[int, str, str]] = []
    for book in payload.get("results") or []:
        if str(book.get("media_type") or "Text") != "Text":
            continue
        try:
            book_id = int(book["id"])
        except (KeyError, TypeError, ValueError):
            continue
        title = str(book.get("title") or f"pg{book_id}")
        url = None
        for key, value in (book.get("formats") or {}).items():
            if not str(key).startswith("text/plain"):
                continue
            if not isinstance(value, str) or not value:
                continue
            if "charset=utf-8" in str(key):
                url = value
                break
            if url is None:
                url = value
        if url:
            out.append((book_id, title, url))
    return out


def repetition_ratio(text: str, n: int = 10) -> float:
    'Gopher-style duplication signal: the share of words inside repeated\n    n-grams plus the share of duplicate lines (0 = fully diverse).'
    words = _WORD_RE.findall(text.lower())
    if len(words) < n * 2:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return 0.0
        return 1.0 - len(set(lines)) / len(lines)
    grams: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        key = tuple(words[i : i + n])
        grams[key] = grams.get(key, 0) + 1
    duplicated = sum((c - 1) * n for c in grams.values() if c > 1)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line_dup = (1.0 - len(set(lines)) / len(lines)) if lines else 0.0
    return min(1.0, 0.5 * duplicated / len(words) + 0.5 * line_dup)


def parse_fineweb_questions(raw: str | None) -> list[tuple[str, str]]:
    'Parse the FineWeb2-HQ-QA ``questions`` field into ``(q, a)`` pairs.\n\n    field is plain text, one pair per line (``- Question: X Answer: Y``);\n    unparsable lines are dropped, never fatal.'
    if not raw:
        return []
    pairs: list[tuple[str, str]] = []
    for line in raw.splitlines():
        hit = _LINE_RE.search(line.strip())
        if hit:
            q = hit.group(1).strip()
            a = hit.group(2).strip()
            if q and a:
                pairs.append((q, a))
    return pairs


def answer_grounded(answer: str, text: str, min_share: float = 0.5) -> bool:
    'Lenient content-word grounding of a QA answer in the source text.\n\n    hallucinated QA pairs without punishing paraphrases.'
    words = [w.lower() for w in _WORD_RE.findall(answer) if len(w) > 2]
    if not words:
        return True
    lowered = text.lower()
    hit = sum(1 for w in words if w in lowered)
    return hit / len(words) >= min_share


def strip_gutenberg_boilerplate(text: str) -> str:
    'Drop the Project Gutenberg license header/footer of a ``.txt`` book.\n\n    the original text when no markers are found (some mirrors strip them).'
    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if start is None and _GUT_START_RE.match(line.strip()):
            start = i + 1
        elif _GUT_END_RE.match(line.strip()):
            end = i
    if start is None and end is None:
        return text.strip()
    if start is None:
        start = 0
    if end is None or end <= start:
        end = len(lines)
    return "\n".join(lines[start:end]).strip()


def looks_human_made(text: str, min_punct_ratio: float = 0.02) -> bool:
    "Human-made proxy for conversational text (mirror of the builder's).\n\n    ≥50 % sentence-initial capitals: auto-generated subtitles fail both."
    stripped = text.strip()
    if not stripped:
        return False
    letters = sum(ch.isalpha() for ch in stripped)
    punct = sum(ch in r'.,!?;:—-()«»"\()' for ch in stripped)
    if letters == 0 or punct / letters < min_punct_ratio:
        return False
    sentences = [s.strip() for s in re.split(r"[.!?]+", stripped) if s.strip()]
    if not sentences:
        return False
    capitalized = sum(
        1 for s in sentences if s[0].isupper() or not s[0].isalpha()
    )
    return capitalized / len(sentences) >= 0.5


def doc_hash(text: str) -> str:
    'Stable short document hash (dedup + doc_id material).'
    return hashlib.blake2b(
        re.sub(r"\s+", " ", text.lower()).encode("utf-8"), digest_size=16
    ).hexdigest()


def make_chunk_record(
    text: str,
    style: str,
    lang: str | None,
    source: str,
    doc_id: str | None = None,
    ontology: str | None = None,
    code_switched: bool = False,
) -> dict[str, Any]:
    """Normalize one document into the builder's chunk-record format."""
    return {
        "text": text.strip(),
        "style": style,
        "lang": lang if lang in ("ru", "en") else detect_lang(text),
        "source": source,
        "doc_id": doc_id or doc_hash(text),
        "ontology": ontology,
        "code_switched": bool(code_switched),
    }


def write_jsonl(path: str | Path, records: Iterator[dict[str, Any]] | list) -> int:
    'Write records as JSONL; returns the count'
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def passes_doc_filters(
    text: str,
    min_words: int,
    max_words: int,
    max_repeat_ratio: float,
) -> bool:
    'Common document-level guards'
    n = count_words(text)
    if n < min_words or n > max_words:
        return False
    if repetition_ratio(text) > max_repeat_ratio:
        return False
    return True


# ---------------------------------------------------------------------------
# Source collectors
# ---------------------------------------------------------------------------
def _iter_parquet_rows(
    repo_id: str, patterns: tuple[str, ...], scan_rows: int
) -> Iterator[dict[str, Any]]:
    'Yield the rows of the hub parquet shards WITHOUT the fsspec streaming.\n\n    ``datasets.load_dataset(streaming=True)``\n    Deterministic: sorted shard names, row order within a shard.'
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "The hub parquet reader needs `huggingface_hub` + `pyarrow` (pip install -e '.[train]')."
        ) from exc
    files = sorted(HfApi().list_repo_files(repo_id, repo_type="dataset"))
    for filename in files:
        if scan_rows <= 0:
            return
        if not any(fnmatch.fnmatch(filename, p) for p in patterns):
            continue
        local = hf_hub_download(repo_id, filename, repo_type="dataset")
        reader = pq.ParquetFile(local)
        for batch in reader.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                yield row
                scan_rows -= 1
                if scan_rows <= 0:
                    return


def collect_wikipedia(args: argparse.Namespace) -> int:
    'Sample a wikimedia/wikipedia snapshot per style/lang/CS quotas.\n\n    (``hf_hub_download`` + pyarrow) — see :func:`_iter_parquet_rows`.'
    rng = random.Random(f"{args.seed}:wikipedia")
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    title_re = re.compile(args.title_regex) if args.title_regex else None
    cs_quota = round(args.code_switch_share * args.max_docs)
    plain_quota = args.max_docs - cs_quota
    taken = {"plain": 0, "cs": 0}

    def records() -> Iterator[dict[str, Any]]:
        style_cycle = list(styles)
        seen_hashes: set[str] = set()
        scanned = 0
        for lang in args.langs.split(","):
            lang = lang.strip()
            patterns = (f"{args.snapshot}.{lang}/train-*.parquet",)
            rows = _iter_parquet_rows(
                "wikimedia/wikipedia", patterns, args.scan_rows - scanned
            )
            for row in rows:
                scanned += 1
                if taken["plain"] + taken["cs"] >= args.max_docs:
                    return
                title = str(row.get("title") or "").strip()
                text = str(row.get("text") or "").strip()
                if title_re and not title_re.search(title):
                    continue
                if not passes_doc_filters(
                    text, args.min_words, args.max_words, args.max_repeat_ratio
                ):
                    continue
                switched = is_code_switched(text, args.cs_min, args.cs_max)
                if switched:
                    if taken["cs"] >= cs_quota:
                        continue
                elif taken["plain"] >= plain_quota:
                    continue
                h = doc_hash(text)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                style = style_cycle[
                    (taken["plain"] + taken["cs"]) % len(style_cycle)
                ]
                yield make_chunk_record(
                    text,
                    style,
                    lang.strip(),
                    f"wikipedia/{lang.strip()}/{title}",
                    code_switched=switched,
                )
                taken["cs" if switched else "plain"] += 1

    n = write_jsonl(args.out, records())
    print(
        f"wikipedia: {n} documents -> {args.out} "
        f"(plain {taken['plain']}, code-switched {taken['cs']})"
    )
    return 0


def collect_gutenberg(args: argparse.Namespace) -> int:
    """Local .txt books -> literary chunk records (boilerplate stripped)."""
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(
            f'--input-dir {root!r} is not a directory.\n--input-dir {root!r}'
        )
    rng = random.Random(f"{args.seed}:gutenberg")

    def records() -> Iterator[dict[str, Any]]:
        files = sorted(root.glob("*.txt"))
        rng.shuffle(files)
        seen_hashes: set[str] = set()
        taken = 0
        for path in files:
            if taken >= args.max_docs:
                break
            raw = path.read_text(encoding="utf-8", errors="replace")
            text = strip_gutenberg_boilerplate(raw)
            if not passes_doc_filters(
                text, args.min_words, args.max_words, args.max_repeat_ratio
            ):
                continue
            h = doc_hash(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            yield make_chunk_record(
                text, "literary", None, f"gutenberg/{path.name}"
            )
            taken += 1

    n = write_jsonl(args.out, records())
    print(f"gutenberg: {n} documents -> {args.out}")
    return 0


def _fetch_text(url: str, timeout_s: float, retries: int = 2) -> str | None:
    'GET a URL as UTF-8 text (``None`` on permanent/per-request failure).\n\n    The request carries ``_USER_AGENT`` (Gutendex and gutenberg.org return\n    403 to the default python-urllib agent).'
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:  # noqa: S310 - fixed https hosts
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError:
            return None
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            # HTTPException covers IncompleteRead/timeout mid-body — a torn
            # transfer is retried, not fatal.
            
            
            if attempt >= retries:
                return None
            time.sleep(min(5.0, 2.0**attempt))
    return None


def _fetch_json(url: str, timeout_s: float) -> dict[str, Any] | None:
    'GET a JSON document'
    raw = _fetch_text(url, timeout_s=timeout_s)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def collect_gutenberg_download(args: argparse.Namespace) -> int:
    'Gutendex listing -> gutenberg.org downloads -> literary chunks.\n\n    ``ru,en``).'
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    title_re = re.compile(args.title_regex) if args.title_regex else None
    workers = max(1, args.workers)
    taken = 0
    failed = 0
    seen_hashes: set[str] = set()

    def _load_book(book_id: int, text_url: str) -> str | None:
        """Read from the cache or download (thread-safe via distinct files)."""
        cache_path = cache / f"pg{book_id}.txt"
        if cache_path.is_file():
            return cache_path.read_text(encoding="utf-8", errors="replace")
        # The Gutendex-provided URL redirects to the ACTUAL file (the
        # cache-path naming varies by upload age: pg<id>.txt vs <id>-0.txt);
        # the fixed cache path is the fallback.
        
        
        
        raw = _fetch_text(text_url, timeout_s=args.timeout)
        if raw is None:
            raw = _fetch_text(
                _GUTEN_CACHE_TPL.format(book_id=book_id),
                timeout_s=args.timeout,
            )
        if raw is None:
            return None
        cache_path.write_text(raw, encoding="utf-8")
        time.sleep(args.sleep)  # per-worker politeness pause
        return raw

    def _process_window(
        window: list[tuple[int, str, str]], out_fh, seen_hashes: set[str]
    ) -> None:
        'Download the window in parallel, apply filters IN CANDIDATE ORDER.'
        nonlocal taken, failed
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_load_book, book_id, text_url)
                for book_id, _title, text_url in window
            ]
            for (book_id, _title, _url), future in zip(window, futures):
                raw = future.result()
                if raw is None:
                    failed += 1
                    continue
                text = strip_gutenberg_boilerplate(raw)
                if not passes_doc_filters(
                    text,
                    args.min_words,
                    args.max_words,
                    args.max_repeat_ratio,
                ):
                    continue
                h = doc_hash(text)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                if taken >= args.max_docs:
                    break  # budget reached mid-window (the rest is cached)
                out_fh.write(
                    json.dumps(
                        make_chunk_record(
                            text,
                            "literary",
                            None,
                            f"gutenberg/pg{book_id}",
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                taken += 1

    with open(args.out, "w", encoding="utf-8") as out_fh:
        for lang in langs:
            if taken >= args.max_docs:
                break
            page_url = f"{_GUTENDEX_URL}?languages={lang}&mime_type=text&page=1"
            pages = 0
            while page_url and taken < args.max_docs and pages < args.max_pages:
                payload = _fetch_json(page_url, timeout_s=args.timeout)
                if not payload:
                    print(f"gutenberg-download: Gutendex page failed: {page_url}")
                    break
                pages += 1
                time.sleep(args.sleep)
                candidates = [
                    (book_id, title, text_url)
                    for book_id, title, text_url in gutendex_candidates(payload)
                    if not (title_re and not title_re.search(title))
                ]
                window: list[tuple[int, str, str]] = []
                for candidate in candidates:
                    if taken >= args.max_docs:
                        break
                    window.append(candidate)
                    if len(window) >= workers * 4:
                        _process_window(window, out_fh, seen_hashes)
                        window = []
                if window:
                    _process_window(window, out_fh, seen_hashes)
                page_url = payload.get("next")
    print(
        f"gutenberg-download: {taken} books -> {args.out}; "
        f"{failed} failed downloads"
    )
    return 0


def collect_ted(args: argparse.Namespace) -> int:
    'Operator-supplied TED transcripts -> conversational chunk records.'
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(
            f'--input {input_path!r} is not a file.\n--input {input_path!r}'
        )
    rng = random.Random(f"{args.seed}:ted")

    def records() -> Iterator[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(input_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rng.shuffle(rows)
        seen_hashes: set[str] = set()
        taken = 0
        for row in rows:
            if taken >= args.max_docs:
                break
            text = str(row.get("transcript") or row.get("text") or "").strip()
            # Conversational text must pass the human-made proxy.
            if not looks_human_made(text):
                continue
            if not passes_doc_filters(
                text, args.min_words, args.max_words, args.max_repeat_ratio
            ):
                continue
            h = doc_hash(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            yield make_chunk_record(
                text,
                "conversational",
                row.get("lang"),
                f"ted/{row.get('url') or row.get('title') or doc_hash(text)}",
                doc_id=f"ted:{h[:16]}",
            )
            taken += 1

    n = write_jsonl(args.out, records())
    print(f"ted: {n} documents -> {args.out}")
    return 0


def collect_fineweb2(args: argparse.Namespace) -> int:
    'Sample the epfml/FineWeb2-HQ-QA rus_Cyrl shards: chunks + QA pairs.\n\n:func:`_iter_parquet_rows`\n\n    ``quality_score``/``language_score``/``pii_count``/``minhash_cluster_size``'
    rng = random.Random(f"{args.seed}:fineweb2")

    def iter_rows():
        yield from _iter_parquet_rows(
            "epfml/FineWeb2-HQ-QA",
            ("data/rus_Cyrl/*.parquet",),
            args.scan_rows,
        )

    def doc_filters(meta: dict[str, Any], text: str) -> bool:
        if meta.get("pii_count", 0):
            return False
        if float(meta.get("language_score", 1.0)) < args.min_language_score:
            return False
        if float(meta.get("quality_score", 0.0)) < args.min_quality_score:
            return False
        if int(meta.get("minhash_cluster_size", 0) or 0) > args.max_minhash_cluster:
            return False
        if cyrillic_share(text) < args.min_cyrillic_share:
            return False
        return passes_doc_filters(
            text, args.min_words, args.max_words, args.max_repeat_ratio
        )

    chunks_path = Path(args.out)
    qa_path = Path(args.qa_out) if args.qa_out else None
    cs_quota = round(args.code_switch_share * args.max_docs)
    plain_quota = args.max_docs - cs_quota
    taken = {"plain": 0, "cs": 0}
    n_docs = n_qa = 0
    seen_hashes: set[str] = set()
    with open(chunks_path, "w", encoding="utf-8") as chunks_fh, (
        open(qa_path, "w", encoding="utf-8") if qa_path else _DevNull()
    ) as qa_fh:
        for row in iter_rows():
            if n_docs >= args.max_docs:
                break
            text = str(row.get("text") or "").strip()
            meta_raw = (row.get("metadata") or {}).get("source_metadata_json")
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except json.JSONDecodeError:
                meta = {}
            if not doc_filters(meta, text):
                continue
            switched = is_code_switched(text, args.cs_min, args.cs_max)
            if switched:
                if taken["cs"] >= cs_quota:
                    continue
            elif taken["plain"] >= plain_quota:
                continue
            h = doc_hash(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            doc_id = str(row.get("id") or h)
            url = str((row.get("metadata") or {}).get("url") or "")
            chunks_fh.write(
                json.dumps(
                    make_chunk_record(
                        text,
                        "journalistic",
                        "ru",
                        "fineweb2/" + doc_id,
                        code_switched=switched,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            taken["cs" if switched else "plain"] += 1
            n_docs += 1
            if qa_path:
                pairs = parse_fineweb_questions(
                    (row.get("metadata") or {}).get("questions")
                )
                rng.shuffle(pairs)
                for q, a in pairs[: args.max_qa_per_doc]:
                    if args.require_grounding and not answer_grounded(a, text):
                        continue
                    if cyrillic_share(q + " " + a) < args.min_cyrillic_share:
                        continue
                    qa_fh.write(
                        json.dumps(
                            {
                                "text": text,
                                "question": q,
                                "answer": a,
                                "source": f"fineweb2/{doc_id}",
                                "url": url,
                                "doc_id": doc_id,
                                "lang": "ru",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_qa += 1
    print(
        f"fineweb2: {n_docs} documents -> {chunks_path} "
        f"(plain {taken['plain']}, code-switched {taken['cs']}); "
        f"{n_qa} QA pairs -> {qa_path}"
    )
    return 0


class _DevNull:
    """A no-op context manager for the optional QA output file."""

    def write(self, _s: str) -> None:  # noqa: D102
        return None

    def close(self) -> None:  # noqa: D102
        return None

    def __enter__(self) -> "_DevNull":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def collect_nerel_bio(args: argparse.Namespace) -> int:
    """GitHub nerel-ds/NEREL-BIO: train chunks + dev/test gold records.

    train → chunk records (``ontology: nerel_bio``; gold annotations are
    deliberately IGNORED — the line is label-free); dev/test → gold records
    in the list format via the brat parser (the frozen anchor/eval material;
    these documents NEVER enter training — gold hygiene).
    """
    from llm_training.extraction.brat import parse_brat_file

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tar_path = cache / f"nerel-bio-{args.ref}.tar.gz"
    if tar_path.is_file():
        # A torn download from an interrupted run is unusable — validate the
        # gzip stream and re-download when it is truncated.
        
        
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.getmembers()
        except (tarfile.TarError, EOFError, OSError):
            print(f"stale/corrupt cached tarball, re-downloading: {tar_path}")
            tar_path.unlink()
    if not tar_path.is_file():
        url = _BIO_TARBALL_URL.format(ref=args.ref)
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, tar_path)  # noqa: S310 - fixed https host
    extract_root = cache / f"nerel-bio-{args.ref}"

    def _find_data_dir(root: Path) -> Path | None:
        """Locate the ``nerel-bio-v1.0`` data dir (with the train split)."""
        candidates = sorted(root.glob(f"*/{_BIO_SUBDIR}")) + sorted(
            root.glob(_BIO_SUBDIR)
        )
        for candidate in candidates:
            if (candidate / "train").is_dir():
                return candidate
        return None

    data_dir = _find_data_dir(extract_root) if extract_root.is_dir() else None
    if data_dir is None:
        # Missing or PARTIAL extraction (an interrupted run leaves a torn
        # tree) — start clean and re-extract.
        
        
        if extract_root.exists():
            shutil.rmtree(extract_root)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_root)  # noqa: S202 - pinned tarball
        data_dir = _find_data_dir(extract_root)
    if data_dir is None:
        raise SystemExit(
            f'nerel-bio-v1.0 not found under {extract_root}'
        )

    n_chunks = n_gold = 0
    with open(args.out, "w", encoding="utf-8") as chunks_fh, open(
        args.gold_out, "w", encoding="utf-8"
    ) as gold_fh:
        for split in ("train", "dev", "test"):
            split_dir = data_dir / split
            if not split_dir.is_dir():
                continue
            for txt in sorted(split_dir.glob("*.txt")):
                text = txt.read_text(encoding="utf-8", errors="replace").strip()
                if not passes_doc_filters(
                    text, args.min_words, args.max_words, args.max_repeat_ratio
                ):
                    continue
                doc_id = f"nerel-bio/{split}/{txt.stem}"
                if split == "train":
                    chunks_fh.write(
                        json.dumps(
                            make_chunk_record(
                                text,
                                "scientific",
                                "ru",
                                f"nerel-bio/{split}/{txt.name}",
                                doc_id=doc_id,
                                ontology="nerel_bio",
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_chunks += 1
                else:
                    ann = txt.with_suffix(".ann")
                    doc = parse_brat_file(ann) if ann.is_file() else None
                    gold_fh.write(
                        json.dumps(
                            {
                                "doc_id": doc_id,
                                "text": text,
                                "ontology": "nerel_bio",
                                "gold": {
                                    "entities": doc.entities if doc else [],
                                    "relations": doc.relations if doc else [],
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_gold += 1
    print(
        f"nerel-bio: {n_chunks} train documents -> {args.out}; "
        f"{n_gold} gold documents -> {args.gold_out}"
    )
    return 0


def parse_lang_shares(spec: str) -> dict[str, float]:
    'Parse ``ru=0.7,en=0.3`` into shares summing to 1.0.'
    shares: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lang, sep, value = part.partition("=")
        if not sep:
            raise SystemExit(
                f'--lang-share: expected lang=share, got {part!r}'
            )
        try:
            share = float(value)
        except ValueError as exc:
            raise SystemExit(
                f'--lang-share: {value!r} is not a number.\n--lang-share: {value!r}'
            ) from exc
        if share <= 0.0:
            raise SystemExit(
                f"--lang-share: share of {lang.strip()!r} must be > 0."
            )
        shares[lang.strip()] = share
    if not shares:
        raise SystemExit(
            '--lang-share is empty'
        )
    total = sum(shares.values())
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(
            f'--lang-share: shares must sum to 1.0, got {total}'
        )
    return shares


def mix_corpus(args: argparse.Namespace) -> int:
    'Combine chunk files into a language-balanced corpus (deterministic).\n\n    wins); each language is subsampled to ``share × N`` where ``N`` is\n    bounded by the SCARCEST language (proportional-to-min quotas), so no\n    language is exhausted; records pass through UNCHANGED in the original\n    file order. Prints the per-language, code-switch and per-source report.'
    shares = parse_lang_shares(args.lang_share)
    by_lang: dict[str, list[dict[str, Any]]] = {lang: [] for lang in shares}
    seen: set[str] = set()
    per_source: dict[str, dict[str, int]] = {}
    for path in args.input:
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f'{path}:{line_no}: not valid JSON ({exc}).\n{path}:{line_no}'
                    ) from exc
                lang = str(
                    rec.get("lang") or detect_lang(str(rec.get("text") or ""))
                )
                if lang not in by_lang:
                    by_lang[lang] = []
                h = doc_hash(str(rec.get("text") or ""))
                if h in seen:
                    continue
                seen.add(h)
                by_lang[lang].append(rec)
                src = str(rec.get("source") or path).split("/", 1)[0]
                per_source.setdefault(src, {}).setdefault(lang, 0)
                per_source[src][lang] += 1

    # Proportional-to-min quotas: no language is exhausted (the scarce one
    # keeps all its documents, the abundant one gives up the excess).
    
    if not any(by_lang.values()):
        raise SystemExit(
            'mix: all inputs are empty'
        )
    scale = min(
        (len(by_lang[lang]) / shares[lang]) for lang in shares if by_lang[lang]
    )
    if args.max_total > 0 and sum(bool(by_lang[l]) for l in shares) == len(shares):
        # The cap applies only when EVERY share language is available —
        # otherwise the missing languages would silently eat the budget.
        
        
        scale = min(scale, float(args.max_total))
    n_total = 0
    with open(args.out, "w", encoding="utf-8") as out_fh:
        for lang in shares:
            avail = by_lang.get(lang) or []
            if not avail:
                print(f"mix: WARNING language {lang!r}: 0 documents available")
                continue
            quota = min(len(avail), round(scale * shares[lang]))
            order = list(range(len(avail)))
            random.Random(f"{args.seed}:mix:{lang}").shuffle(order)
            picked = sorted(order[:quota])
            for i in picked:
                out_fh.write(json.dumps(avail[i], ensure_ascii=False) + "\n")
            n_cs = sum(1 for i in picked if avail[i].get("code_switched"))
            n_total += quota
            print(
                f"mix: {lang}: {quota} of {avail and len(avail)} documents "
                f"(share {shares[lang]:.2f}, code-switched {n_cs})"
            )
    for src in sorted(per_source):
        counts = ", ".join(
            f"{lang}={per_source[src][lang]}" for lang in sorted(per_source[src])
        )
        print(f"mix: source {src}: {counts}")
    print(f"mix: {n_total} documents -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# CLI / CLI
# ---------------------------------------------------------------------------
def _add_common_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-words", type=int, default=40)
    p.add_argument("--max-words", type=int, default=20000)
    p.add_argument("--max-repeat-ratio", type=float, default=0.3)
    p.add_argument("--max-docs", type=int, default=10000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Collect the open sources of the GRPO-extraction corpus.'
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_wiki = sub.add_parser("wikipedia", help="HF wikimedia/wikipedia snapshot")
    p_wiki.add_argument("--out", required=True)
    p_wiki.add_argument("--snapshot", default="20231101")
    p_wiki.add_argument("--langs", default="ru,en")
    p_wiki.add_argument(
        "--styles",
        default="scientific,journalistic",
        help='Round-robin style assignment',
    )
    p_wiki.add_argument("--title-regex", default=None)
    p_wiki.add_argument("--scan-rows", type=int, default=1_000_000)
    p_wiki.add_argument(
        "--code-switch-share",
        type=float,
        default=0.0,
        help=(
            'Share of the budget reserved for RU-EN code-switched documents (the IT register).'
        ),
    )
    p_wiki.add_argument(
        "--cs-min",
        type=float,
        default=0.03,
        help='Min Latin-word share of the code-switch band',
    )
    p_wiki.add_argument(
        "--cs-max",
        type=float,
        default=0.5,
        help='Max Latin-word share of the code-switch band',
    )
    _add_common_filters(p_wiki)

    p_gut = sub.add_parser("gutenberg", help="local .txt books (public domain)")
    p_gut.add_argument("--out", required=True)
    p_gut.add_argument("--input-dir", required=True)
    _add_common_filters(p_gut)

    p_gdl = sub.add_parser(
        "gutenberg-download",
        help="gutenberg.org .txt via the Gutendex listing (public domain)",
    )
    p_gdl.add_argument("--out", required=True)
    p_gdl.add_argument(
        "--langs",
        default="ru,en",
        help='Sequential language quotas from one budget',
    )
    p_gdl.add_argument("--title-regex", default=None)
    p_gdl.add_argument(
        "--cache-dir",
        default="/tmp/opencode/gutenberg_cache",
        help='Download cache (re-runs resume)',
    )
    p_gdl.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help='Seconds between requests (politeness)',
    )
    p_gdl.add_argument(
        "--workers",
        type=int,
        default=6,
        help=(
            'Parallel download threads'
        ),
    )
    p_gdl.add_argument("--timeout", type=float, default=60.0)
    p_gdl.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help='Gutendex page cap per language',
    )
    _add_common_filters(p_gdl)

    p_ted = sub.add_parser("ted", help="operator-supplied transcripts")
    p_ted.add_argument("--out", required=True)
    p_ted.add_argument("--input", required=True)
    _add_common_filters(p_ted)

    p_fw2 = sub.add_parser("fineweb2", help="epfml/FineWeb2-HQ-QA rus_Cyrl")
    p_fw2.add_argument("--out", required=True)
    p_fw2.add_argument("--qa-out", default=None)
    p_fw2.add_argument("--scan-rows", type=int, default=1_000_000)
    p_fw2.add_argument("--min-quality-score", type=float, default=0.5)
    p_fw2.add_argument("--min-language-score", type=float, default=0.9)
    p_fw2.add_argument("--max-minhash-cluster", type=int, default=10)
    p_fw2.add_argument("--min-cyrillic-share", type=float, default=0.6)
    p_fw2.add_argument("--max-qa-per-doc", type=int, default=3)
    p_fw2.add_argument(
        "--require-grounding",
        action="store_true",
        default=False,
        help='Lenient answer-in-text check',
    )
    p_fw2.add_argument(
        "--code-switch-share",
        type=float,
        default=0.0,
        help=(
            'Share of the budget reserved for RU-EN code-switched documents (the IT register).'
        ),
    )
    p_fw2.add_argument("--cs-min", type=float, default=0.03)
    p_fw2.add_argument("--cs-max", type=float, default=0.5)
    _add_common_filters(p_fw2)

    p_bio = sub.add_parser("nerel-bio", help="github nerel-ds/NEREL-BIO v1.0")
    p_bio.add_argument("--out", required=True)
    p_bio.add_argument("--gold-out", required=True)
    p_bio.add_argument("--ref", default="master")
    p_bio.add_argument("--cache-dir", default="/tmp/opencode/nerel_bio_cache")
    _add_common_filters(p_bio)

    p_mix = sub.add_parser(
        "mix",
        help="combine chunk files with exact language quotas (deterministic)",
    )
    p_mix.add_argument("--out", required=True)
    p_mix.add_argument(
        "--input",
        action="append",
        required=True,
        help='Chunk JSONL file (repeatable)',
    )
    p_mix.add_argument(
        "--lang-share",
        default="ru=0.7,en=0.3",
        help='Target language shares',
    )
    p_mix.add_argument(
        "--max-total",
        type=int,
        default=0,
        help='Total cap (0 = unlimited)',
    )
    p_mix.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    handlers = {
        "wikipedia": collect_wikipedia,
        "gutenberg": collect_gutenberg,
        "gutenberg-download": collect_gutenberg_download,
        "ted": collect_ted,
        "fineweb2": collect_fineweb2,
        "nerel-bio": collect_nerel_bio,
        "mix": mix_corpus,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
