"Offline tests of the TED transcript downloader's pure helpers (no network)."

from __future__ import annotations

import json

import pytest

from scripts.download_ted import (
    clean_cue_text,
    enumerate_talk_slugs,
    extract_next_data,
    talk_slug_from_loc,
    transcript_from_next_data,
)


def _page(payload: dict) -> str:
    return (
        "<html><head></head><body><script>"
        f'<script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(payload, ensure_ascii=False)}</script>"
        "</script></body></html>"
    )


def _talk_payload(paragraphs, lang="ru", title="Выступление", url="https://www.ted.com/talks/x"):
    return {
        "props": {
            "pageProps": {
                "videoData": {"title": title, "canonicalUrl": url},
                "transcriptData": {
                    "translation": {
                        "language": {"internalLanguageCode": lang},
                        "paragraphs": paragraphs,
                    }
                },
            }
        }
    }


def test_talk_slug_from_loc() -> None:
    assert (
        talk_slug_from_loc("https://www.ted.com/talks/bill_gates_the_next_outbreak")
        == "bill_gates_the_next_outbreak"
    )
    # Non-talk pages are rejected
    assert talk_slug_from_loc("https://www.ted.com/talks") is None
    assert (
        talk_slug_from_loc("https://www.ted.com/tedx/events/1234/x") is None
    )
    assert talk_slug_from_loc("https://www.ted.com/playlists/57/x") is None


def test_clean_cue_text() -> None:
    assert clean_cue_text("Поэтому в подвалах\nу всех были бочки") == (
        "Поэтому в подвалах у всех были бочки"
    )
    assert clean_cue_text("A <i>bold</i> <br/>claim &amp; more") == "A bold claim & more"
    assert clean_cue_text("   ") == ""
    # Speaker markers are prose to keep (filters handle them)
    assert clean_cue_text("(Laughter)") == "(Laughter)"


def test_extract_next_data() -> None:
    payload = {"props": {"pageProps": {}}}
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(payload)
        + "</script>"
    )
    assert extract_next_data(html) == payload
    assert extract_next_data("<html>no data</html>") is None
    assert extract_next_data(
        '<script id="__NEXT_DATA__" type="application/json">{broken</script>'
    ) is None


def test_transcript_from_next_data_ru() -> None:
    payload = _talk_payload(
        [
            {"cues": [
                {"text": "Во времена моего детства", "time": 5504},
                {"text": "мы боялись\n<i>ядерной</i> войны.", "time": 7398},
            ]},
            {"cues": [{"text": "Поэтому &amp; мы действуем.", "time": 12000}]},
        ]
    )
    row = transcript_from_next_data(extract_next_data(_page(payload)))
    assert row is not None
    assert row["lang"] == "ru"
    assert row["title"] == "Выступление"
    assert row["url"] == "https://www.ted.com/talks/x"
    assert row["transcript"].split("\n")[0] == (
        "Во времена моего детства мы боялись ядерной войны."
    )
    assert row["transcript"].split("\n")[1] == "Поэтому & мы действуем."


def test_transcript_from_next_data_fallback_lang() -> None:
    # No ru translation -> the site returns the original; the row carries
    # the ACTUAL returned language
    
    payload = _talk_payload(
        [{"cues": [{"text": "Hello world.", "time": 100}]}], lang="en"
    )
    row = transcript_from_next_data(extract_next_data(_page(payload)))
    assert row is not None and row["lang"] == "en"


def test_transcript_from_next_data_empty() -> None:
    # No transcript at all
    payload = {"props": {"pageProps": {"transcriptData": {"translation": {
        "language": {"internalLanguageCode": "ru"}, "paragraphs": [],
    }}}}}
    assert transcript_from_next_data(extract_next_data(_page(payload))) is None
    assert transcript_from_next_data(None) is None
    assert transcript_from_next_data({"props": {"pageProps": {}}}) is None


def test_enumerate_talk_slugs_rejects_non_talk(monkeypatch) -> None:
    archive = (
        "<urlset>"
        "<url><loc>https://www.ted.com/talks/bbb</loc></url>"
        "<url><loc>https://www.ted.com/talks/aaa</loc></url>"
        "<url><loc>https://www.ted.com/talks/aaa</loc></url>"  # duplicate
        "<url><loc>https://www.ted.com/tedx/events/1/x</loc></url>"
        "<url><loc>https://www.ted.com/playlists/57/x</loc></url>"
        "</urlset>"
    )

    def fake_fetch(url: str, timeout_s: float, retries: int = 2):
        import gzip as _gzip

        assert url.endswith("videos.xml.gz")
        return _gzip.compress(archive.encode())

    monkeypatch.setattr("scripts.download_ted._fetch", fake_fetch)
    slugs = enumerate_talk_slugs(timeout_s=1.0)
    assert slugs == ["aaa", "bbb"]


def test_enumerate_talk_slugs_no_archives(monkeypatch) -> None:
    import gzip as _gzip

    def fake_fetch(url: str, timeout_s: float, retries: int = 2):
        return _gzip.compress(b"")

    monkeypatch.setattr("scripts.download_ted._fetch", fake_fetch)
    with pytest.raises(SystemExit):
        enumerate_talk_slugs(timeout_s=1.0)
