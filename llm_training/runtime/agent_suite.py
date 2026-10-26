"Frozen agent-suite validation: policy rollout + deterministic scoring.\n\n\nStage 3 of ``specifications/2026-10-17-agent-suite-glm53-validation.md``\n(the FOURTH validation element): at ctrl-points the training model solves\nthe frozen suite tasks through the SGR protocol (reasoning -> ``searchdocs``\nover the frozen index -> work -> final ``finalanswertool``) and every task\nis scored by DETERMINISTIC metrics — no judge, no network:\n\n* ``envelope`` — the run ends with a valid ``finalanswertool`` envelope\n  (non-empty answer, known status);\n* ``schema`` — every ``searchdocs`` call carried valid arguments\n  (query 8–256 chars, ``top_k`` int 1–10);\n* ``retrieval`` — the share of gold content stems found in the union of\n  the retrieved chunk texts (pseudo-gold entity names / summary\n  descriptions / QA reference);\n* ``grounding`` — the share of the final answer's content stems present in\n  the retrieved chunks (no invented content);\n* ``answer`` — family-specific correctness against the gold: extraction =\n  canonical entity-name F1 vs the pseudo-gold; qa = normalized exact match\n  vs the reference; summary = the share of the reference-description stems\n  reproduced in the answer description.\n\nTask score = strict harmonic mean of the active metrics (one zero kills the\ntask — the strong-gradient convention); ``agent_val`` = mean over tasks.\nThe selection key at anchors is ``harmonic(val, agent_val)``: see\n:func:`selection_key` for the fallback rules (a failed rollout or an\nunder-covered pool rolls the key back to ``val`` alone — a measurement\nfailure is never treated as degradation, the drift-probe rule).\n\n``agent_val``\n``harmonic(val, agent_val)``"

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Callable

from llm_training.extraction.retrieval import HybridIndex, tokenize_stems
from llm_training.reward.scoring import harmonic_mean

#: The suite row format version this module reads.

FORMAT_VERSION: int = 1

#: Turn cap of the lockstep policy rollout (the fixed schedule: reasoning ->
#: searchdocs -> work -> final; the force-final nudge precedes the last
# : turn).

DEFAULT_MAX_TURNS: int = 4

#: Mini-batch of the batched per-turn generation.

DEFAULT_MINI_BATCH: int = 8

#: Per-turn generation cap (tool calls / reasoning / the final envelope).

DEFAULT_MAX_NEW_TOKENS: int = 2048

#: Retrieval recall window (the spec's recall@10).

DEFAULT_RECALL_K: int = 10

#: Share of suite tasks that must be scored for agent_val to enter the
#: selection key (below the floor the key rolls back to val).

AGENT_VAL_FLOOR: float = 0.9

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_FINAL_TOOL = "finalanswertool"
_SEARCH_TOOL = "searchdocs"


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    'Tolerant parse of ``<tool_call>`` blocks (fences and prose tolerated).\n\n    payloads are skipped, never raised — a broken turn just scores poorly.'
    import json as _json

    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        raw = match.group(1).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        try:
            payload = _json.loads(raw)
        except _json.JSONDecodeError:
            decoder = _json.JSONDecoder()
            try:
                payload, _ = decoder.raw_decode(raw)
            except _json.JSONDecodeError:
                continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or "").strip()
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = _json.loads(arguments)
            except _json.JSONDecodeError:
                arguments = {}
        if name and isinstance(arguments, dict):
            calls.append({"name": name, "arguments": arguments})
    return calls


def validate_searchdocs_args(arguments: dict[str, Any]) -> float:
    'Schema check of one searchdocs call: 1.0 valid, 0.0 broken.\n\n    Contract: ``query`` string 8–256 chars; ``top_k`` int 1–10.'
    query = arguments.get("query")
    top_k = arguments.get("top_k")
    ok = isinstance(query, str) and 8 <= len(query) <= 256
    ok = ok and isinstance(top_k, int) and not isinstance(top_k, bool)
    ok = ok and 1 <= top_k <= 10
    return 1.0 if ok else 0.0


def content_stems(text: str) -> set[str]:
    "Content stems of a text (len > 2) — the probe's target tokens."
    return {stem for stem in tokenize_stems(text) if len(stem) > 2}


def _answer_text(final: dict[str, Any] | None) -> str:
    'The final answer as plain text (JSON payloads are serialized).'
    if not final:
        return ""
    answer = final.get("answer")
    if not isinstance(answer, str):
        answer = json.dumps(answer or "", ensure_ascii=False)
    return answer


class AgentSuiteScorer:
    'The frozen suite: lockstep policy rollout + deterministic scoring.\n\n    Built once per training run; :meth:`evaluate` is called at every anchor\n    with the CURRENT policy weights.'

    def __init__(
        self,
        *,
        suite_dir: str | Path,
        corpus_paths: list[str],
        local_embed_model: str = "BAAI/bge-m3",
        embed_device: str = "cuda",
        embed_base_url: str | None = None,
        embed_model_id: str | None = None,
        embed_api_key: str = "EMPTY",
        encode: Callable[[list[str]], Any] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        mini_batch: int = DEFAULT_MINI_BATCH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        recall_k: int = DEFAULT_RECALL_K,
    ) -> None:
        self.suite_dir = Path(suite_dir)
        self.rows: list[dict[str, Any]] = []
        with (self.suite_dir / "suite.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise SystemExit(
                f'{self.suite_dir}: empty suite.jsonl.\n{self.suite_dir}'
            )
        digest_file = self.suite_dir / "index" / "digest.txt"
        if not digest_file.exists():
            raise SystemExit(
                f'{digest_file}missing — the frozen index digest pins the instrument.'
            )
        self.index_digest = digest_file.read_text(encoding="utf-8").strip()
        if encode is not None:
            # Injectable encoder (offline tests, alternate backends).
            
            self._encode = encode
        elif embed_base_url and embed_model_id:
            from llm_training.extraction.retrieval import OpenAIEmbeddingsEncoder

            self._encode = OpenAIEmbeddingsEncoder(
                base_url=embed_base_url,
                model_id=embed_model_id,
                api_key=embed_api_key,
            )
        else:
            from sentence_transformers import SentenceTransformer  # lazy, heavy.

            model = SentenceTransformer(local_embed_model, device=embed_device)

            def _encode(texts: list[str]) -> Any:
                return model.encode(
                    list(texts), batch_size=16, show_progress_bar=False
                )

            self._encode = _encode
        self.index = HybridIndex.load(
            self.suite_dir / "index", encode=self._encode
        )
        self.texts_by_id: dict[str, str] = {}
        for path in corpus_paths:
            corpus = Path(path)
            if not corpus.exists():
                continue
            with corpus.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row_id = str(row.get("id") or "")
                    text = str(
                        ((row.get("extra") or {}).get("extraction") or {}).get(
                            "text"
                        )
                        or ""
                    ).strip()
                    if row_id and text:
                        self.texts_by_id[row_id] = text
        self.max_turns = max(2, int(max_turns))
        self.mini_batch = max(1, int(mini_batch))
        self.max_new_tokens = max(64, int(max_new_tokens))
        self.recall_k = max(1, int(recall_k))
        self.n_tasks = len(self.rows)

    # ------------------------------------------------------------- rollout

    def rollout(
        self,
        generate_batch: Callable[[list[list[dict[str, str]]]], list[str]],
        rows: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        'Lockstep policy rollout: all tasks advance turn-by-turn together.\n\n        and returns one response per task (the model adapter hides the\n        chat template and the greedy decode). ``rows`` — an optional task\n        subset (the rank-sharded anchor rollout); ``None`` — all tasks.\n\n        ``generate_batch``'
        task_rows = self.rows if rows is None else rows
        states: list[dict[str, Any]] = []
        for row in task_rows:
            states.append(
                {
                    "row": row,
                    "messages": [
                        {"role": m["role"], "content": m["content"]}
                        for m in row["prompt"]
                    ],
                    "done": False,
                    "final": None,
                    "turns": 0,
                    "search_calls": 0,
                    "schema_flags": [],
                    "retrieved_ids": [],
                    "forced": False,
                    "error": None,
                }
            )

        for turn in range(1, self.max_turns + 1):
            active = [s for s in states if not s["done"]]
            if not active:
                break
            force_now = turn >= self.max_turns and not all(
                s["forced"] for s in active
            )
            for s in active:
                if force_now and not s["forced"]:
                    s["forced"] = True
                    s["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                "STOP searching. Emit the finalanswertool "
                                "envelope NOW with the material you have "
                                "already collected. Do not call searchdocs "
                                "again."
                            ),
                        }
                    )
            for start in range(0, len(active), self.mini_batch):
                batch = active[start : start + self.mini_batch]
                responses = generate_batch([s["messages"] for s in batch])
                for state, text in zip(batch, responses, strict=True):
                    state["turns"] = turn
                    for call in extract_tool_calls(text):
                        name = call["name"]
                        arguments = call["arguments"]
                        if name == _SEARCH_TOOL:
                            state["schema_flags"].append(
                                validate_searchdocs_args(arguments)
                            )
                            state["search_calls"] += 1
                            query = str(arguments.get("query") or "").strip()
                            if query:
                                hits = self.index.search(
                                    query, top_k=self.recall_k
                                )
                                for hit in hits:
                                    if hit.doc_id not in state["retrieved_ids"]:
                                        state["retrieved_ids"].append(
                                            hit.doc_id
                                        )
                            state["messages"].append(
                                {"role": "assistant", "content": text}
                            )
                            enriched = [
                                {
                                    "doc_id": doc_id,
                                    "text": self.texts_by_id.get(doc_id, "")[
                                        :1200
                                    ],
                                }
                                for doc_id in state["retrieved_ids"]
                            ]
                            state["messages"].append(
                                {
                                    "role": "user",
                                    "content": "[tool] searchdocs result:\n"
                                    + json.dumps(enriched, ensure_ascii=False),
                                }
                            )
                            state.setdefault("served_this_turn", True)
                            break
                        if name == _FINAL_TOOL:
                            answer = arguments.get("answer")
                            status = str(arguments.get("status") or "")
                            state["final"] = {
                                "reasoning": str(arguments.get("reasoning") or ""),
                                "completed_steps": arguments.get("completed_steps")
                                or [],
                                "answer": answer,
                                "status": status,
                            }
                            state["done"] = True
                            break
                    else:
                        # Neither tool: nudge (the model drifted off protocol).
                        
                        state["messages"].append(
                            {"role": "assistant", "content": text}
                        )
                        state["messages"].append(
                            {
                                "role": "user",
                                "content": (
                                    "Emit a <tool_call> with name searchdocs, "
                                    "or the finalanswertool envelope."
                                ),
                            }
                        )
                    if name == _SEARCH_TOOL and state.get("served_this_turn"):
                        state["served_this_turn"] = False
        records = []
        for state in states:
            row = state["row"]
            final = state["final"]
            accepted = (
                final is not None
                and bool(final.get("answer"))
                and final.get("status") in {"completed", "failed"}
            )
            records.append(
                {
                    "suite_id": row["id"],
                    "family": row["family"],
                    "source_row_id": row["source_row_id"],
                    "task_class": row.get("task_class"),
                    "accepted_envelope": accepted,
                    "turns": state["turns"],
                    "search_calls": state["search_calls"],
                    "schema_flags": state["schema_flags"],
                    "retrieved_ids": state["retrieved_ids"],
                    "final": final,
                    "messages": state["messages"],
                }
            )
        return records

    # --------------------------------------------------------------- score

    def score(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        'Deterministic per-task scoring + the agent_val breakdown.'
        rows_by_id = {row["id"]: row for row in self.rows}
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        family_sums: dict[str, list[float]] = {}
        task_scores: list[float] = []
        for record in records:
            row = rows_by_id[record["suite_id"]]
            gold = row.get("gold") or {}
            family = str(row.get("family"))
            retrieved = " ".join(
                self.texts_by_id.get(doc_id, "")
                for doc_id in record["retrieved_ids"]
            )
            retrieved_stems = content_stems(retrieved)
            answer_text = _answer_text(record["final"])
            answer_stems = content_stems(answer_text)

            envelope = 1.0 if record["accepted_envelope"] else 0.0
            schema = (
                sum(record["schema_flags"]) / len(record["schema_flags"])
                if record["schema_flags"]
                else 0.0
            )
            if family == "extraction":
                target_text = " ".join(
                    str(e.get("entity_name") or "")
                    for e in (
                        (gold.get("pseudo_gold") or {}).get("entities") or []
                    )
                    if isinstance(e, dict)
                )
            elif family == "summary":
                material = gold.get("material") or {}
                parts = [
                    str(material.get("entity_name") or material.get("subject") or "")
                ]
                parts.extend(str(d) for d in (material.get("descriptions") or [])[:8])
                target_text = " ".join(parts)
            else:
                target_text = str(gold.get("reference") or "")
            target_stems = content_stems(target_text)
            recall = (
                len(target_stems & retrieved_stems) / len(target_stems)
                if target_stems
                else 0.0
            )
            grounding = (
                len(answer_stems & retrieved_stems) / len(answer_stems)
                if answer_stems
                else 0.0
            )
            answer = self._answer_score(family, record, gold)
            active = [envelope, schema, recall, grounding, answer]
            task_score = harmonic_mean(active)
            task_scores.append(task_score)
            for key, value in (
                ("envelope", envelope),
                ("schema", schema),
                ("recall", recall),
                ("grounding", grounding),
                ("answer", answer),
            ):
                metric_sums[key] = metric_sums.get(key, 0.0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1
            family_sums.setdefault(family, []).append(task_score)

        breakdown: dict[str, Any] = {
            "agent_val": (
                sum(task_scores) / len(task_scores) if task_scores else 0.0
            ),
            "n_scored": len(task_scores),
            "envelope_rate": metric_sums["envelope"] / max(1, metric_counts["envelope"]),
            "schema_rate": metric_sums["schema"] / max(1, metric_counts["schema"]),
            "recall": metric_sums["recall"] / max(1, metric_counts["recall"]),
            "grounding": metric_sums["grounding"] / max(1, metric_counts["grounding"]),
            "answer": metric_sums["answer"] / max(1, metric_counts["answer"]),
            "index_digest": self.index_digest,
        }
        breakdown["per_family"] = {
            family: {
                "agent_val": round(sum(values) / len(values), 4),
                "n": len(values),
            }
            for family, values in sorted(family_sums.items())
        }
        return breakdown

    def _answer_score(
        self, family: str, record: dict[str, Any], gold: dict[str, Any]
    ) -> float:
        'Family-specific correctness against the gold (deterministic).'
        answer_text = _answer_text(record["final"]).strip()
        if not answer_text:
            return 0.0
        if family == "extraction":
            from llm_training.reward.extraction import (
                parse_extraction_output,
                pseudo_gold_score,
            )

            scored = pseudo_gold_score(
                parse_extraction_output(answer_text),
                gold.get("pseudo_gold"),
                None,
                gold.get("text"),
            )
            if scored is None or scored.combined is None:
                return 0.0
            return float(scored.combined)
        if family == "qa":
            reference = content_stems(str(gold.get("reference") or ""))
            if not reference:
                return 0.0
            answer_stems = content_stems(answer_text)
            overlap = len(reference & answer_stems) / len(reference)
            exact = 1.0 if answer_text.strip() == str(gold.get("reference") or "").strip() else 0.0
            return max(exact, 0.5 * overlap + 0.5 * exact)
        # summary: the share of the reference-description stems reproduced.
        reference = (gold.get("reference") or {})
        ref_text = ""
        if isinstance(reference, dict):
            ref_text = str(reference.get("description") or "") or str(
                reference.get("response") or ""
            )
        ref_stems = content_stems(ref_text)
        if not ref_stems:
            return 0.0
        answer_stems = content_stems(answer_text)
        return len(ref_stems & answer_stems) / len(ref_stems)

    # ------------------------------------------------------------ evaluate

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        generate_batch_fn: Callable[
            [list[list[dict[str, str]]]], list[str]
        ]
        | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
        'Rollout + score the CURRENT policy (the anchor-time entry).\n\n        ``generate_batch_fn`` (message lists → responses) routes generation\n        through the colocate vLLM engine; with it and ``world_size > 1`` the\n        tasks are sharded across ranks and the records gathered to rank 0\n        (non-zero ranks return ``(None, None)``). Without it the HF path\n        generates on ``model`` as before. Returns ``(breakdown, records)``;\n        raises on infrastructure failures — the CALLER maps a failure to the\n        val-only key fallback.'
        import torch

        engine_mode = generate_batch_fn is not None
        if not engine_mode:
            was_training = model.training
            model.eval()
            prev_padding = tokenizer.padding_side
            tokenizer.padding_side = "left"
            prev_use_cache = getattr(model.config, "use_cache", None)
            model.config.use_cache = True

        def generate_batch(message_lists: list[list[dict[str, str]]]) -> list[str]:
            from llm_training.tokenizer_utils import decode_generation

            if generate_batch_fn is not None:
                return generate_batch_fn(message_lists)
            prompts = [
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in message_lists
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            trimmed = outputs[:, inputs["input_ids"].shape[1] :]
            return [
                decode_generation(tokenizer, row) for row in trimmed
            ]

        try:
            with torch.no_grad():
                shard_rows = (
                    [self.rows[i] for i in range(rank, len(self.rows), world_size)]
                    if world_size > 1 and generate_batch_fn is not None
                    else self.rows
                )
                records = self.rollout(generate_batch, rows=shard_rows)
            if world_size > 1 and generate_batch_fn is not None:
                gathered = _gather_objects(records, world_size)
                if rank != 0:
                    return None, None
                records = [rec for sub in gathered for rec in sub]
            breakdown = self.score(records)
            return breakdown, records
        finally:
            if not engine_mode:
                tokenizer.padding_side = prev_padding
                if prev_use_cache is not None:
                    model.config.use_cache = prev_use_cache
                if was_training:
                    model.train()


def _gather_objects(obj: Any, world_size: int) -> list[Any]:
    '``all_gather_object`` under an initialized process group; else raise.\n\n    ``all_gather_object``'
    import torch

    if world_size <= 1 or not (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ):
        raise RuntimeError(
            f'torch.distributed is not initialized, but a multi-rank agent-suite gather was requested (world_size={world_size}'
        )
    gathered: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(gathered, obj)
    return gathered


def selection_key(
    val: float,
    agent_val: float | None,
    n_scored: int,
    n_total: int,
    floor: float = AGENT_VAL_FLOOR,
) -> tuple[float, bool]:
    'The anchor selection key: ``harmonic(val, agent_val)`` with fallbacks.\n\n    (spec section 9): a failed rollout (``agent_val is None``) or an\n    under-covered pool (``n_scored < floor * n_total``) rolls the key back\n    to ``val`` alone — a measurement failure is never degradation (the\n    drift-probe rule). Returns ``(key, agent_used)``.'
    covered = n_total > 0 and n_scored >= floor * n_total
    if agent_val is None or not covered:
        return val, False
    key = harmonic_mean([val, agent_val])
    return (key if key is not None else val), True
