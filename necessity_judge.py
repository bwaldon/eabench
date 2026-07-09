"""LLM-backed necessity judge for CFR (regulation) sources — issue #7.

The deterministic ``analyze_sources.py`` computes objective *signals*; it cannot
render the underlying legal judgment of whether a regulation is actually
*required* to answer an item. This module asks an LLM (OpenAI by default,
following the repo's ``eval_pipeline.py`` convention of ``gpt-5.4``) to make that
call per CFR source, returning a structured verdict + rationale.

Design goals:

* **Reproducible & cheap.** Results are cached to JSON keyed by a fingerprint of
  the exact inputs + rubric version + model, so re-running is free unless an
  input changed or ``--refresh`` is passed.
* **Offline-testable.** The network call is injected as a ``respond`` callable;
  tests pass a fake responder, so no key or spend is needed to exercise the
  parsing/caching/validation logic.
* **Defensive.** The model is asked for strict JSON; output is validated against
  the expected shape and one retry is attempted before giving up on an item.

This judge *augments* the deterministic signals — it does not replace them.
``analyze_sources.py`` places verdict and signal side by side and flags
disagreements for human review. No benchmark items are modified.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent
DEFAULT_CACHE = ROOT / "necessity_judge_results.json"
DEFAULT_MODEL = "gpt-5.4"  # matches eval_pipeline.py

# Bump when the rubric/prompt changes so cached verdicts are invalidated.
RUBRIC_VERSION = "1"

VERDICTS = ("ESSENTIAL", "REDUNDANT", "REVIEW")

Responder = Callable[[list[dict]], str]

_RUBRIC = """\
You are a FERPA legal analyst auditing a benchmark of question/answer items.
Each item is grounded in one or more legal SOURCES: statutes (20 U.S.C. § 1232g)
and/or regulations (34 C.F.R. part 99). We are auditing whether the CFR
(regulation) sources are actually needed.

For EACH regulation source listed, decide whether it is:

- "ESSENTIAL": the regulation supplies an operative rule that the correct answer
  depends on and that is NOT already provided by a statute source in this item.
  Removing it would make the item unanswerable or strip the decisive authority.
- "REDUNDANT": the regulation merely restates authority that a statute source in
  this item already provides, AND the answer does not depend on the regulation
  specifically. It could be deleted without changing the correct answer.
- "REVIEW": genuinely borderline, tangential, or you cannot tell — a human
  should decide.

Judge only what the item's own text supports. Be strict: prefer REDUNDANT/REVIEW
over ESSENTIAL when the statute alone would yield the same answer.

Return ONLY a JSON object with this exact shape:
{
  "item_verdict": "ESSENTIAL|REDUNDANT|REVIEW",   // overall for the item's CFR usage
  "cfr_sources": [
    {"location": "<verbatim location string>",
     "verdict": "ESSENTIAL|REDUNDANT|REVIEW",
     "rationale": "<=200 chars"}
  ]
}
Include one cfr_sources entry per regulation source, echoing its location exactly."""


@dataclass
class JudgeResult:
    task_id: str
    item_verdict: str
    cfr_verdicts: dict[str, dict]  # location -> {"verdict", "rationale"}
    model: str
    fingerprint: str

    def to_json(self) -> dict:
        return {
            "task_id": self.task_id,
            "item_verdict": self.item_verdict,
            "cfr_verdicts": self.cfr_verdicts,
            "model": self.model,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_json(cls, d: dict) -> "JudgeResult":
        return cls(
            task_id=d["task_id"],
            item_verdict=d["item_verdict"],
            cfr_verdicts=d["cfr_verdicts"],
            model=d["model"],
            fingerprint=d["fingerprint"],
        )


def _cfr_sources(record: dict) -> list[dict]:
    return [s for s in record.get("source", []) if s["source_type"] == "regulation"]


def fingerprint(record: dict, model: str) -> str:
    """Stable hash of everything that could change a verdict."""
    cfr = _cfr_sources(record)
    payload = {
        "rubric": RUBRIC_VERSION,
        "model": model,
        "user_query": record.get("user_query", ""),
        "query_continuation": record.get("query_continuation", ""),
        "expected_final_answer": record.get("expected_final_answer", ""),
        "sources": [
            {"location": s["location"], "type": s["source_type"], "text": s["source_text"]}
            for s in record.get("source", [])
        ],
        "cfr_locations": [s["location"] for s in cfr],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_messages(record: dict) -> list[dict]:
    sources_block = "\n\n".join(
        f"SOURCE {i + 1} [{s['source_type']}] {s['location']}\n{s['source_text']}"
        for i, s in enumerate(record.get("source", []))
    )
    cfr = _cfr_sources(record)
    cfr_block = "\n".join(f"- {s['location']}" for s in cfr)
    user = (
        f"ITEM {record['task_id']}\n\n"
        f"USER QUERY:\n{record.get('user_query', '')}\n\n"
        f"QUERY CONTINUATION:\n{record.get('query_continuation', '') or '(none)'}\n\n"
        f"ALL SOURCES:\n{sources_block}\n\n"
        f"EXPECTED FINAL ANSWER:\n{record.get('expected_final_answer', '')}\n\n"
        f"REGULATION SOURCES TO JUDGE:\n{cfr_block}"
    )
    return [
        {"role": "system", "content": _RUBRIC},
        {"role": "user", "content": user},
    ]


class JudgeError(RuntimeError):
    pass


def _parse_and_validate(raw: str, record: dict) -> tuple[str, dict[str, dict]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise JudgeError(f"non-JSON response: {e}") from e
    item_verdict = data.get("item_verdict")
    if item_verdict not in VERDICTS:
        raise JudgeError(f"bad item_verdict {item_verdict!r}")
    entries = data.get("cfr_sources")
    if not isinstance(entries, list) or not entries:
        raise JudgeError("missing cfr_sources")
    expected = {s["location"] for s in _cfr_sources(record)}
    verdicts: dict[str, dict] = {}
    for e in entries:
        loc = e.get("location")
        verdict = e.get("verdict")
        if verdict not in VERDICTS:
            raise JudgeError(f"bad verdict {verdict!r} for {loc!r}")
        verdicts[loc] = {"verdict": verdict, "rationale": str(e.get("rationale", "")).strip()}
    missing = expected - set(verdicts)
    if missing:
        raise JudgeError(f"missing verdicts for {sorted(missing)}")
    return item_verdict, verdicts


def judge_item(record: dict, *, respond: Responder, model: str, retries: int = 1) -> JudgeResult:
    """Judge one item's CFR sources, retrying once on malformed output."""
    messages = build_messages(record)
    last_err: Exception | None = None
    for _ in range(retries + 1):
        raw = respond(messages)
        try:
            item_verdict, verdicts = _parse_and_validate(raw, record)
        except JudgeError as e:
            last_err = e
            continue
        return JudgeResult(
            task_id=record["task_id"],
            item_verdict=item_verdict,
            cfr_verdicts=verdicts,
            model=model,
            fingerprint=fingerprint(record, model),
        )
    raise JudgeError(f"item {record['task_id']}: {last_err}")


def make_openai_responder(model: str, api_key: str | None = None) -> Responder:
    """Build a responder backed by the OpenAI client (lazy import; needs a key)."""
    import openai

    client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def respond(messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    return respond


def _load_cache(path: Path) -> dict[str, JudgeResult]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: JudgeResult.from_json(v) for k, v in raw.items()}


def _save_cache(path: Path, cache: dict[str, JudgeResult]) -> None:
    payload = {k: v.to_json() for k, v in cache.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_judge(
    records: list[dict],
    *,
    respond: Responder,
    model: str = DEFAULT_MODEL,
    cache_path: Path | None = DEFAULT_CACHE,
    refresh: bool = False,
) -> dict[str, JudgeResult]:
    """Judge every CFR-referencing record, using/refreshing the cache.

    Only items whose fingerprint is missing or stale (or all, if ``refresh``)
    trigger a model call. Returns ``{task_id: JudgeResult}``.
    """
    cache = {} if refresh or cache_path is None else _load_cache(cache_path)
    results: dict[str, JudgeResult] = {}
    dirty = False
    for record in records:
        if not _cfr_sources(record):
            continue
        tid = record["task_id"]
        fp = fingerprint(record, model)
        cached = cache.get(tid)
        if cached is not None and cached.fingerprint == fp and cached.model == model:
            results[tid] = cached
            continue
        result = judge_item(record, respond=respond, model=model)
        results[tid] = result
        cache[tid] = result
        dirty = True
    if dirty and cache_path is not None:
        _save_cache(cache_path, cache)
    return results
