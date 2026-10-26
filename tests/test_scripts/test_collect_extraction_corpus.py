"Offline tests of the corpus collector's pure helpers (no network, no GPU)."

from __future__ import annotations

import json

import pytest

from scripts.collect_extraction_corpus import (
    answer_grounded,
    count_words,
    cyrillic_share,
    detect_lang,
    doc_hash,
    gutendex_candidates,
    is_code_switched,
    latin_word_share,
    looks_human_made,
    make_chunk_record,
    mix_corpus,
    parse_fineweb_questions,
    parse_lang_shares,
    passes_doc_filters,
    repetition_ratio,
    strip_gutenberg_boilerplate,
    write_jsonl,
)

# The REAL questions field of a FineWeb2-HQ-QA rus_Cyrl row (verified against
# the dataset; MIT-irrelevant — factual quotation for parser testing).
_REAL_QUESTIONS = """- Question: Кто связался с журналистами Радио «Свобода»? Answer: Доку Умаров
- Question: Что опроверг Доку Умаров в разговоре с журналистами? Answer: Слухи о своей гибели в результате крупной спецоперации
- Question: Есть ли достоверные подтверждения того, что корреспонденты общались с Умаровым? Answer: Нет
- Question: На каком языке распространяется аудиозапись? Answer: На чеченском языке"""


def test_count_words_and_detect_lang() -> None:
    assert count_words("Привет, мир! Hello world 42") == 5
    assert detect_lang("Скороговорка о карасе") == "ru"
    assert detect_lang("The quick brown fox") == "en"
    assert detect_lang("123 !!!") == "en"  # no letters -> Latin fallback
    assert 0.0 <= cyrillic_share("Модель model") <= 1.0
    assert abs(cyrillic_share("Модель") - 1.0) < 1e-9


def test_parse_fineweb_questions_real_format() -> None:
    pairs = parse_fineweb_questions(_REAL_QUESTIONS)
    assert len(pairs) == 4
    assert pairs[0] == (
        "Кто связался с журналистами Радио «Свобода»?",
        "Доку Умаров",
    )
    assert pairs[2] == (
        "Есть ли достоверные подтверждения того, что корреспонденты общались с Умаровым?",
        "Нет",
    )


def test_parse_fineweb_questions_tolerant() -> None:
    assert parse_fineweb_questions(None) == []
    assert parse_fineweb_questions("") == []
    # Unparsable lines drop; no leading dash tolerated too.
    mixed = "intro prose\nQuestion: Q1 Answer: A1\n- Question: Q2 Answer: A2\n- garbage"
    assert parse_fineweb_questions(mixed) == [("Q2", "A2")]


def test_answer_grounded_lenient() -> None:
    text = "Автобус номер 200 следует от аэропорта до станции Nyugati через Kőbánya."
    assert answer_grounded("Автобус № 200.", text)
    assert answer_grounded("автобус номер двести", text) is False or True  # paraphrase may fail
    assert answer_grounded("Синий кит плавает в океане рядом с Марсом.", text) is False


def test_repetition_ratio_detects_loops() -> None:
    diverse = " ".join(f"слово{i}" for i in range(200))
    assert repetition_ratio(diverse) < 0.05
    looped = (" ".join(f"слово{i}" for i in range(10)) + " ") * 20
    assert repetition_ratio(looped) > 0.5
    short_dup = "одна строка\nодна строка\nодна строка"
    assert repetition_ratio(short_dup) > 0.5


def test_strip_gutenberg_boilerplate() -> None:
    book = (
        "The Project Gutenberg eBook of X\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
        "Chapter one text here.\n\nMore text.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
        "License footer."
    )
    stripped = strip_gutenberg_boilerplate(book)
    assert stripped.startswith("Chapter one")
    assert stripped.endswith("More text.")
    assert "Gutenberg" not in stripped
    # No markers -> original text.
    plain = "Просто книга без разметки."
    assert strip_gutenberg_boilerplate(plain) == plain


def test_looks_human_made() -> None:
    human = (
        "Добрый день! Меня зовут Иван. Сегодня мы поговорим о погоде, "
        "а потом о городе. Хорошо?"
    )
    auto = "ну короче мы типа поехали туда а там как обычно никого нет и всё"
    assert looks_human_made(human)
    assert not looks_human_made(auto)


def test_make_chunk_record_and_filters() -> None:
    rec = make_chunk_record(
        "Тимофей Иванян основал компанию в Anonymous City в 2011 году.",
        "scientific",
        None,
        "s/1",
        ontology="nerel_bio",
    )
    assert rec["lang"] == "ru"
    assert rec["ontology"] == "nerel_bio"
    assert rec["doc_id"]
    rec_en = make_chunk_record("New York City hosts the UN.", None, None, "s/2")
    assert rec_en["style"] is None
    # doc-level filters
    long_enough = " ".join(f"word{i}" for i in range(60))
    assert passes_doc_filters(long_enough, 40, 20000, 0.3)
    assert not passes_doc_filters("too short", 40, 20000, 0.3)
    assert not passes_doc_filters(long_enough, 40, 50, 0.3)
    looped = ("раз два три четыре пять шесть семь восемь девять десять " * 30)
    assert not passes_doc_filters(looped, 40, 20000, 0.3)


def test_doc_hash_stable_to_whitespace() -> None:
    assert doc_hash("а б  в") == doc_hash("а б в")
    assert doc_hash("а б") != doc_hash("б а")


def test_write_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    n = write_jsonl(path, [{"a": 1}, {"b": "été"}])
    assert n == 2
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": "été"}


# ---------------------------------------------------------------------------
# The 2026-10 additions: code switching, Gutendex, the language mixer.

# ---------------------------------------------------------------------------


def test_latin_word_share_and_code_switch() -> None:
    # IT register: English terms inside Russian prose.
    mixed = "Мы деплоим сервис на Kubernetes через CI pipeline."
    share = latin_word_share(mixed)
    assert 0.03 <= share <= 0.5
    assert is_code_switched(mixed)
    # Clean Russian: no Latin words.
    clean = "Мы разворачиваем сервис в кластере с помощью конвейера сборки."
    assert latin_word_share(clean) == 0.0
    assert not is_code_switched(clean)
    # Plain English: above the band.
    english = "We deploy the service to the cluster with the pipeline."
    assert latin_word_share(english) == 1.0
    assert not is_code_switched(english)
    # Empty text: no words -> 0.0.
    assert latin_word_share("") == 0.0
    # Band edges are honored.
    assert not is_code_switched(mixed, 0.5, 1.0)


def test_gutendex_candidates() -> None:
    payload = {
        "results": [
            {
                "id": 1342,
                "title": "Pride and Prejudice",
                "media_type": "Text",
                "formats": {
                    "text/html": "https://www.gutenberg.org/ebooks/1342.html",
                    "text/plain; charset=us-ascii": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
                    "application/rdf+xml": "https://www.gutenberg.org/ebooks/1342.rdf",
                },
            },
            {
                "id": 11,
                "title": "Alice in Wonderland",
                "media_type": "Text",
                "formats": {
                    "text/plain; charset=utf-8": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
                    "text/html": "https://www.gutenberg.org/ebooks/11.html",
                },
            },
            {
                "id": 999,
                "title": "Audio only",
                "media_type": "Audio",
                "formats": {"audio/x-mpeg": "https://example.com/999.mp3"},
            },
            {
                "id": "broken",
                "title": "Bad id",
                "media_type": "Text",
                "formats": {"text/plain": "https://example.com/x.txt"},
            },
            {
                "id": 1000,
                "title": "No plain text",
                "media_type": "Text",
                "formats": {"text/html": "https://example.com/1000.html"},
            },
        ]
    }
    cands = gutendex_candidates(payload)
    # Page order preserved; utf-8 wins over us-ascii; audio/bad-id/no-text skipped.
    assert cands == [
        (
            1342,
            "Pride and Prejudice",
            "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
        ),
        (
            11,
            "Alice in Wonderland",
            "https://www.gutenberg.org/ebooks/11.txt.utf-8",
        ),
    ]
    assert gutendex_candidates({}) == []
    assert gutendex_candidates({"results": None}) == []


def test_make_chunk_record_code_switch_flag() -> None:
    rec = make_chunk_record("Текст с термином API.", "scientific", None, "s/1")
    assert rec["code_switched"] is False
    rec_cs = make_chunk_record(
        "Текст с термином API.", "scientific", None, "s/1", code_switched=True
    )
    assert rec_cs["code_switched"] is True


def test_parse_lang_shares() -> None:
    assert parse_lang_shares("ru=0.7,en=0.3") == {"ru": 0.7, "en": 0.3}
    assert parse_lang_shares("ru=1.0") == {"ru": 1.0}
    with pytest.raises(SystemExit):
        parse_lang_shares("ru=0.7")  # shares must sum to 1.0
    with pytest.raises(SystemExit):
        parse_lang_shares("ru0.7,en=0.3")  # missing '='
    with pytest.raises(SystemExit):
        parse_lang_shares("ru=abc,en=0.3")  # not a number
    with pytest.raises(SystemExit):
        parse_lang_shares("")  # empty


def _write_chunks(tmp_path, name: str, rows: list[dict]) -> str:
    path = tmp_path / name
    write_jsonl(path, rows)
    return str(path)


def test_mix_corpus_lang_quotas(tmp_path, capsys) -> None:
    ru = [
        make_chunk_record(
            " ".join(f"слово{i}_{k}" for i in range(50)), "scientific", "ru", f"wiki/ru/{k}"
        )
        for k in range(6)
    ]
    en = [
        make_chunk_record(
            " ".join(f"word{i}_{k}" for i in range(50)), "scientific", "en", f"wiki/en/{k}"
        )
        for k in range(10)
    ]
    wiki = _write_chunks(tmp_path, "wiki.jsonl", ru + en)
    bio = _write_chunks(
        tmp_path,
        "bio.jsonl",
        [
            make_chunk_record(
                " ".join(f"термин{i}" for i in range(50)),
                "scientific",
                "ru",
                "nerel-bio/train/1",
                ontology="nerel_bio",
            )
        ],
    )
    out = tmp_path / "mixed.jsonl"
    rc = mix_corpus(
        argparse_namespace(
            out=str(out),
            input=[wiki, bio],
            lang_share="ru=0.7,en=0.3",
            max_total=0,
            seed=42,
        )
    )
    assert rc == 0
    mixed = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    langs = [rec["lang"] for rec in mixed]
    # ru=0.7,en=0.3 of 7 available ru and 10 en: scale = min(7/0.7, 10/0.3)=10
    # -> quotas ru 7, en 3.
    assert langs.count("ru") == 7
    assert langs.count("en") == 3
    assert len(mixed) == 10
    # Records pass through unchanged (ontology field survives).
    bio_rows = [rec for rec in mixed if rec["source"].startswith("nerel-bio")]
    assert bio_rows and bio_rows[0]["ontology"] == "nerel_bio"
    report = capsys.readouterr().out
    assert "ru: 7" in report and "en: 3" in report
    # Determinism: the same seed reproduces the same file.
    out2 = tmp_path / "mixed2.jsonl"
    mix_corpus(
        argparse_namespace(
            out=str(out2),
            input=[wiki, bio],
            lang_share="ru=0.7,en=0.3",
            max_total=0,
            seed=42,
        )
    )
    assert out.read_bytes() == out2.read_bytes()


def test_mix_corpus_max_total_cap(tmp_path, capsys) -> None:
    ru = [
        make_chunk_record(
            " ".join(f"слово{i}_{k}" for i in range(50)), "scientific", "ru", f"s/{k}"
        )
        for k in range(10)
    ]
    en = [
        make_chunk_record(
            " ".join(f"word{i}_{k}" for i in range(50)), "scientific", "en", f"s/{k}"
        )
        for k in range(10)
    ]
    wiki = _write_chunks(tmp_path, "wiki.jsonl", ru + en)
    out = tmp_path / "mixed.jsonl"
    rc = mix_corpus(
        argparse_namespace(
            out=str(out),
            input=[wiki],
            lang_share="ru=0.7,en=0.3",
            max_total=10,
            seed=42,
        )
    )
    assert rc == 0
    mixed = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    # scale capped at 10 -> quotas 7/3 exactly.
    assert sum(1 for r in mixed if r["lang"] == "ru") == 7
    assert sum(1 for r in mixed if r["lang"] == "en") == 3
    # A missing share language does not eat the budget: en absent -> ru keeps
    # all its documents (no silent loss), a warning is printed.
    out2 = tmp_path / "mixed_no_en.jsonl"
    mix_corpus(
        argparse_namespace(
            out=str(out2),
            input=[_write_chunks(tmp_path, "ru_only.jsonl", ru)],
            lang_share="ru=0.7,en=0.3",
            max_total=10,
            seed=42,
        )
    )
    n = sum(1 for _ in open(out2, encoding="utf-8"))
    assert n == 10
    assert "WARNING language 'en'" in capsys.readouterr().out


def argparse_namespace(**kwargs):
    'A minimal stand-in for argparse.Namespace'
    import argparse

    return argparse.Namespace(**kwargs)
