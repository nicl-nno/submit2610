'Brat ``.ann`` parsing for the NEREL-BIO gold conversion (GRPO-extraction).\n\n\nNEREL-BIO distributes its expert annotations in the `brat annotation tool`_\nstandoff format: a ``.txt`` document plus a ``.ann`` sidecar where ``T``\nlines mark entity mentions (with character offsets, possibly fragmented)\nand ``R`` lines mark directed relations as ``Arg1``/``Arg2`` references to\n``T`` ids. This module converts one ``.ann`` file into the LIST format the\nextraction line uses everywhere (``{"entities": [...], "relations": [...]}``\nwith ``{entity_name, entity_type}`` /\n``{source_entity, relation_type, target_entity}`` items) — the same shape\n``parse_extraction_output`` produces and ``pseudo_gold_score`` scores, so\ngold annotations, judge pseudo-gold and policy responses all live in one\ncoordinate system. Offsets are DROPPED (the list format is offset-free);\nnested mentions simply become separate list items (the nested-NER-as-lists\nsemantics both sides share). Duplicate mentions are collapsed into unique\n``(name, type)`` / ``(source, relation, target)`` pairs — the set-F1\nscoring canonicalizes both sides anyway.\nNEREL-BIO\n``T``-line examples\n\n    T1\tNUMBER 135 138\t194\n    R1\tABBREVIATION Arg1:T4 Arg2:T5\n\n.. _brat annotation tool: https://brat.nlplab.org/'

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_ARG1_RE = re.compile(r"\bArg1:(T\d+)\b")
_ARG2_RE = re.compile(r"\bArg2:(T\d+)\b")


@dataclass
class BratDocument:
    'One parsed ``.ann`` sidecar\n\n    Attributes:\n        entities: unique ``{entity_name, entity_type, description}`` items\n            (mention order preserved on first occurrence).\n        relations: unique ``{source_entity, relation_type, target_entity,\n            description}`` items; endpoints resolve to the MENTION TEXT of\n            the referenced ``T`` lines.\n        n_mentions: the raw ``T``-line count (before dedup) — the honest\n            nested-mention volume of the document.\n        skipped: counts of the ignored line kinds (``#``, ``E``, ``A``,\n            ``M``, ``N``, malformed ``R``) — transparency, never an abort.'

    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    n_mentions: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


def parse_brat_ann(text: str) -> BratDocument:
    "Parse one ``.ann`` file's content into the list extraction format."
    mention: dict[str, tuple[str, str]] = {}  # tid -> (name, entity_type)
    relation_lines: list[tuple[str, str]] = []  # (relation_type, args tail)
    skipped: dict[str, int] = {}
    n_mentions = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        kind = line[0]
        if kind == "T":
            n_mentions += 1
            _consume_t_line(line, mention)
        elif kind == "R":
            hit = _parse_r_line(line)
            if hit is None:
                skipped["R_malformed"] = skipped.get("R_malformed", 0) + 1
            else:
                relation_lines.append(hit)
        elif kind in "#EAMN":
            # Annotator notes / events / attributes / mentions /
            # normalizations (UMLS links) — outside the v1 scope.
            skipped[kind] = skipped.get(kind, 0) + 1

    entities: list[dict] = []
    seen_entities: set[tuple[str, str]] = set()
    for name, etype in mention.values():
        if (name, etype) not in seen_entities:
            seen_entities.add((name, etype))
            entities.append(
                {"entity_name": name, "entity_type": etype, "description": ""}
            )

    relations: list[dict] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for rel, args in relation_lines:
        arg1 = _ARG1_RE.search(args)
        arg2 = _ARG2_RE.search(args)
        if not arg1 or not arg2:
            skipped["R_malformed"] = skipped.get("R_malformed", 0) + 1
            continue
        src = mention.get(arg1.group(1))
        tgt = mention.get(arg2.group(1))
        if src is None or tgt is None:
            # A relation referencing a missing/dropped mention.
            skipped["R_dangling"] = skipped.get("R_dangling", 0) + 1
            continue
        key = (src[0], rel, tgt[0])
        if key not in seen_relations:
            seen_relations.add(key)
            relations.append(
                {
                    "source_entity": src[0],
                    "relation_type": rel,
                    "target_entity": tgt[0],
                    "description": "",
                }
            )

    return BratDocument(
        entities=entities,
        relations=relations,
        n_mentions=n_mentions,
        skipped=skipped,
    )


def parse_brat_file(path: str | Path) -> BratDocument:
    'Parse one ``.ann`` file from disk (UTF-8, tolerant).'
    return parse_brat_ann(Path(path).read_text(encoding="utf-8", errors="replace"))


def _consume_t_line(line: str, mention: dict[str, tuple[str, str]]) -> None:
    """Fold one ``T`` line into the mention map; malformed lines drop."""
    parts = line.split("\t")
    if len(parts) < 3:
        return
    tid, body, text = parts[0], parts[1], "\t".join(parts[2:])
    fields = body.split()
    if len(fields) < 2:
        return
    etype = fields[0]
    name = " ".join(text.split())
    if name:
        mention[tid] = (name, etype)


def _parse_r_line(line: str) -> tuple[str, str] | None:
    """Split one ``R`` line into ``(relation_type, args tail)``."""
    parts = line.split("\t")
    if len(parts) < 2:
        return None
    fields = parts[1].split()
    if len(fields) < 3:
        return None
    return fields[0], " ".join(fields[1:])
