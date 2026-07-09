"""Offline tests for necessity_judge (run: ``python test_necessity_judge.py``).

Uses a fake in-process responder — no API key, no network, no spend. Exercises
prompt construction, JSON validation, retry, fingerprinting, and caching.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from necessity_judge import (
    JudgeError,
    build_messages,
    fingerprint,
    judge_item,
    run_judge,
)

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} :: {detail}")
        _failures.append(name)


def S(loc, t, text="text"):
    return {"location": loc, "source_text": text, "source_type": t}


ITEM = {
    "task_id": "38",
    "user_query": "q",
    "query_continuation": "",
    "expected_final_answer": "answer",
    "source": [
        S("20 U.S.C. § 1232g(a)(4)(A)", "statute"),
        S("34 CFR § 99.3", "regulation"),
    ],
}


def responder_for(item_verdict, source_verdicts, *, drop=None, bad_verdict=False):
    """Build a fake responder echoing the CFR locations it is asked to judge."""
    def respond(messages):
        import re

        user = messages[1]["content"]
        locs = re.findall(r"- (34 CFR[^\n]+)", user.split("REGULATION SOURCES TO JUDGE:")[1])
        entries = []
        for loc in locs:
            if drop and loc == drop:
                continue
            v = "NOPE" if bad_verdict else source_verdicts.get(loc, "REVIEW")
            entries.append({"location": loc, "verdict": v, "rationale": "r"})
        return json.dumps({"item_verdict": item_verdict, "cfr_sources": entries})

    return respond


def test_valid_judge() -> None:
    print("test_valid_judge")
    r = judge_item(
        ITEM,
        respond=responder_for("REDUNDANT", {"34 CFR § 99.3": "REDUNDANT"}),
        model="fake",
    )
    check("item verdict", r.item_verdict == "REDUNDANT", r.item_verdict)
    check("source verdict", r.cfr_verdicts["34 CFR § 99.3"]["verdict"] == "REDUNDANT")
    check("only cfr judged", set(r.cfr_verdicts) == {"34 CFR § 99.3"}, str(r.cfr_verdicts))


def test_messages_contain_answer_and_cfr() -> None:
    print("test_messages_contain_answer_and_cfr")
    msgs = build_messages(ITEM)
    check("two messages", len(msgs) == 2)
    check("system has rubric", "ESSENTIAL" in msgs[0]["content"])
    check("user lists cfr to judge", "34 CFR § 99.3" in msgs[1]["content"])
    check("statute source included for context", "1232g(a)(4)(A)" in msgs[1]["content"])


def test_bad_json_raises() -> None:
    print("test_bad_json_raises")
    try:
        judge_item(ITEM, respond=lambda m: "not json", model="fake")
        check("should raise", False)
    except JudgeError:
        check("bad json raises JudgeError", True)


def test_missing_source_verdict_raises() -> None:
    print("test_missing_source_verdict_raises")
    try:
        judge_item(
            ITEM,
            respond=responder_for("REVIEW", {}, drop="34 CFR § 99.3"),
            model="fake",
        )
        check("should raise on missing verdict", False)
    except JudgeError:
        check("missing verdict raises", True)


def test_bad_verdict_value_raises() -> None:
    print("test_bad_verdict_value_raises")
    try:
        judge_item(ITEM, respond=responder_for("REVIEW", {}, bad_verdict=True), model="fake")
        check("should raise on bad verdict", False)
    except JudgeError:
        check("bad verdict value raises", True)


def test_fingerprint_changes_with_inputs() -> None:
    print("test_fingerprint_changes_with_inputs")
    fp1 = fingerprint(ITEM, "gpt-5.4")
    check("stable", fp1 == fingerprint(ITEM, "gpt-5.4"))
    check("model-sensitive", fp1 != fingerprint(ITEM, "other-model"))
    changed = {**ITEM, "expected_final_answer": "different"}
    check("answer-sensitive", fp1 != fingerprint(changed, "gpt-5.4"))


def test_cache_prevents_requery() -> None:
    print("test_cache_prevents_requery")
    calls = {"n": 0}

    def counting(messages):
        calls["n"] += 1
        return responder_for("ESSENTIAL", {"34 CFR § 99.3": "ESSENTIAL"})(messages)

    with TemporaryDirectory() as d:
        cache = Path(d) / "c.json"
        run_judge([ITEM], respond=counting, model="fake", cache_path=cache)
        after_first = calls["n"]
        run_judge([ITEM], respond=counting, model="fake", cache_path=cache)
        check("first run calls model", after_first == 1, str(after_first))
        check("second run served from cache", calls["n"] == after_first, str(calls["n"]))
        # refresh forces a re-query
        run_judge([ITEM], respond=counting, model="fake", cache_path=cache, refresh=True)
        check("refresh re-queries", calls["n"] == after_first + 1, str(calls["n"]))


def test_non_cfr_item_skipped() -> None:
    print("test_non_cfr_item_skipped")
    statute_only = {**ITEM, "source": [S("20 U.S.C. § 1232g(a)(1)(A)", "statute")]}
    out = run_judge([statute_only], respond=lambda m: "x", model="fake", cache_path=None)
    check("statute-only skipped", out == {}, str(out))


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print()
    if _failures:
        print(f"FAILED {len(_failures)} check(s): {_failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
