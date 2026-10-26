'Download the TED talk transcripts (ted.com, offline helper for the collector).\n\n\nCatalog enumeration: the sitemap archive\n``https://www.ted.com/sitemaps/videos.xml.gz`` lists every main-line TED\ntalk (TEDx excluded). Transcripts: the page\n``/talks/<slug>/transcript?language=<lang>`` embeds the cue JSON in\n``__NEXT_DATA__`` (``props.pageProps.transcriptData``); the response carries\nthe ACTUAL language of the returned translation in\n``translation.language.internalLanguageCode``. The site does NOT fall back\nautomatically: a talk without the preferred-language translation returns NO\ntranscript, so each talk is requested TWICE — the preferred language first\n(``ru``), then the English original when the first attempt came back empty;\ntalks with no transcript at all (partner uploads) are skipped as empty.\n\nOutput: the collector\'s TED JSONL format — one row per talk:\n``{"transcript": ..., "title": ..., "url": ..., "lang": "ru"|"en"}``.\nCues are grouped into Paragraphs by the site already; cue texts are joined\ninto prose (HTML tags stripped, entities unescaped, internal line breaks\nfolded), paragraphs are separated by newlines.\n\nResume-ability: each talk\'s parsed record is cached as one JSON file under\n``--cache-dir``; re-runs skip the network for cached slugs. Politeness: one\nrequest per ``--sleep`` seconds (default 1.0), retries with backoff on\ntransient errors.\n\nExamples::\n\n    python scripts/download_ted.py --out ~/datasets/ted/transcripts.jsonl\n    python scripts/download_ted.py --limit 5 --out /tmp/opencode/ted_smoke.jsonl'

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

#: The User-Agent for ted.com (a plain descriptive client, not a browser).
_USER_AGENT = "llm-training-corpus-collector/1.0"

#: The sitemap videos archive (the full main-line talk catalog) /

_VIDEOS_ARCHIVE_URL = "https://www.ted.com/sitemaps/videos.xml.gz"
_TALK_LOC_RE = re.compile(r"^https://www\.ted\.com/talks/([A-Za-z0-9_\-]+)$")

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Pure helpers (no network; unit-tested)
# ---------------------------------------------------------------------------
def talk_slug_from_loc(loc: str) -> str | None:
    'Extract the talk slug from a sitemap ``<loc>`` URL (non-talks -> None).'
    hit = _TALK_LOC_RE.match(loc.strip())
    return hit.group(1) if hit else None


def clean_cue_text(text: str) -> str:
    'Strip HTML tags, unescape entities, fold whitespace'
    stripped = _TAG_RE.sub(" ", html.unescape(text))
    return _WS_RE.sub(" ", stripped).strip()


def extract_next_data(html_text: str) -> dict[str, Any] | None:
    """Parse the ``__NEXT_DATA__`` JSON blob (None when absent/corrupt)."""
    hit = _NEXT_DATA_RE.search(html_text)
    if not hit:
        return None
    try:
        return json.loads(hit.group(1))
    except json.JSONDecodeError:
        return None


def transcript_from_next_data(
    data: dict[str, Any] | None,
) -> dict[str, str] | None:
    'Build the collector row from ``__NEXT_DATA__`` (None when no transcript).\n\n    The language is whatever the site RETURNED (``ru`` when the translation\n    exists, the original otherwise) — the caller filters by ``--langs``.'
    if not data:
        return None
    page_props = data.get("props", {}).get("pageProps", {})
    transcript = page_props.get("transcriptData") or {}
    translation = transcript.get("translation") or {}
    paragraphs = translation.get("paragraphs") or []
    video = page_props.get("videoData") or {}
    lines: list[str] = []
    for paragraph in paragraphs:
        cues = paragraph.get("cues") or []
        text = " ".join(
            cleaned for cue in cues if (cleaned := clean_cue_text(str(cue.get("text") or "")))
        )
        if text:
            lines.append(text)
    if not lines:
        return None
    lang = str(
        (translation.get("language") or {}).get("internalLanguageCode") or ""
    )
    return {
        "transcript": "\n".join(lines),
        "title": str(video.get("title") or ""),
        "url": str(video.get("canonicalUrl") or ""),
        "lang": lang,
    }


# ---------------------------------------------------------------------------
# Networked helpers
# ---------------------------------------------------------------------------
def _fetch(
    url: str, timeout_s: float, retries: int = 2, _redirects: int = 5
) -> bytes | None:
    "GET bytes with UA + retry/backoff (None on permanent failure).\n\n    HTTP 308 follows manually: urllib's redirect handler covers\n    301/302/303/307 but NOT 308, and ted.com redirects its legacy\n    ``/talks/<slug>/transcript`` path with exactly that code."
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:  # noqa: S310 - fixed https hosts
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 308 and _redirects > 0:
                location = exc.headers.get("Location")
                if location:
                    return _fetch(
                        urllib.parse.urljoin(url, location),
                        timeout_s=timeout_s,
                        retries=retries,
                        _redirects=_redirects - 1,
                    )
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(min(30.0, 5.0 * 2**attempt))
                continue
            return None
        except (urllib.error.URLError, OSError):
            if attempt >= retries:
                return None
            time.sleep(min(10.0, 2.0**attempt))
    return None


def _fetch_text(url: str, timeout_s: float) -> str | None:
    raw = _fetch(url, timeout_s=timeout_s)
    return raw.decode("utf-8", errors="replace") if raw is not None else None


def enumerate_talk_slugs(timeout_s: float) -> list[str]:
    'List every main-line talk slug from the sitemap videos archive.'
    raw = _fetch(_VIDEOS_ARCHIVE_URL, timeout_s=timeout_s)
    if raw is None:
        raise SystemExit(
            f'sitemap videos archive unavailable: {_VIDEOS_ARCHIVE_URL}'
        )
    xml = gzip.decompress(raw).decode("utf-8", errors="replace")
    loc_re = re.compile(r"<loc>\s*([^<]+?)\s*</loc>")
    slugs = sorted(
        {slug for loc in loc_re.findall(xml) if (slug := talk_slug_from_loc(loc))}
    )
    if not slugs:
        raise SystemExit(
            'no talk slugs in the sitemap videos archive.'
        )
    return slugs


def _cache_path(cache_dir: Path, slug: str) -> Path:
    digest = hashlib.blake2b(slug.encode("utf-8"), digest_size=8).hexdigest()
    return cache_dir / f"{digest}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Download the TED talk transcripts from ted.com.'
        )
    )
    parser.add_argument(
        "--out",
        default=str(Path.home() / "datasets/ted/transcripts.jsonl"),
        help='Output JSONL',
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/opencode/ted_cache",
        help='Per-talk cache (re-runs resume)',
    )
    parser.add_argument(
        "--langs",
        default="ru,en",
        help='Acceptable transcript languages',
    )
    parser.add_argument(
        "--prefer-lang",
        default="ru",
        help='Requested language first',
    )
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help='Talk cap for a smoke run (0 = all)',
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    accepted_langs = {lang.strip() for lang in args.langs.split(",") if lang.strip()}

    slugs = enumerate_talk_slugs(timeout_s=args.timeout)
    if args.limit > 0:
        slugs = slugs[: args.limit]
    print(f"talks enumerated: {len(slugs)}")

    n_written = n_cached = n_empty = n_failed = n_foreign = n_en_tried = 0
    by_lang: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as out_fh:
        for index, slug in enumerate(slugs, 1):
            cached = _cache_path(cache_dir, slug)
            if cached.is_file():
                row = json.loads(cached.read_text(encoding="utf-8"))
                n_cached += 1
            else:
                page = _fetch_text(
                    f"https://www.ted.com/talks/{slug}/transcript"
                    f"?language={args.prefer_lang}",
                    timeout_s=args.timeout,
                )
                row = transcript_from_next_data(extract_next_data(page or ""))
                if row is None and (
                    "en" in accepted_langs and args.prefer_lang != "en"
                ):
                    # The site does NOT fall back automatically: a talk
                    # without the preferred-language translation returns NO
                    # transcript — ask for the English original explicitly.
                    
                    
                    page = _fetch_text(
                        f"https://www.ted.com/talks/{slug}/transcript"
                        f"?language=en",
                        timeout_s=args.timeout,
                    )
                    row = transcript_from_next_data(extract_next_data(page or ""))
                    n_en_tried += 1
                if row is None:
                    n_empty += 1
                    time.sleep(args.sleep)
                    continue
                cached.write_text(
                    json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                time.sleep(args.sleep)
            lang = row["lang"]
            if lang not in accepted_langs:
                n_foreign += 1
                continue
            row = {
                "transcript": row["transcript"],
                "title": row["title"],
                "url": row["url"] or f"https://www.ted.com/talks/{slug}",
                "lang": lang,
            }
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            by_lang[lang] = by_lang.get(lang, 0) + 1
            n_written += 1
            if index % 200 == 0:
                print(f"... {index}/{len(slugs)} (written {n_written})")
    print(
        f"ted: {n_written} transcripts -> {out_path} "
        f"(by lang {by_lang}; cached {n_cached}, empty {n_empty}, "
        f"foreign lang {n_foreign}, en retried {n_en_tried}, "
        f"failed {len(slugs) - n_written - n_empty - n_foreign})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
