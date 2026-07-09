"""Tests for analyze_sources (run: ``python test_analyze_sources.py``).

Dependency-free; validates the necessity *signals* on synthetic records so the
logic is pinned independently of the live benchmark content.
"""

from __future__ import annotations

from analyze_sources import analyze_item, summarize

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} :: {detail}")
        _failures.append(name)


def rec(task_id, sources, answer=""):
    return {"task_id": task_id, "source": sources, "expected_final_answer": answer}


def S(loc, t):
    return {"location": loc, "source_text": "", "source_type": t}


def test_non_cfr_item_is_skipped() -> None:
    print("test_non_cfr_item_is_skipped")
    a = analyze_item(rec("1", [S("20 U.S.C. § 1232g(a)(1)(A)", "statute")]))
    check("statute-only returns None", a is None)


def test_mixed_item_signals() -> None:
    print("test_mixed_item_signals")
    a = analyze_item(
        rec(
            "30",
            [S("20 U.S.C. § 1232g(b)(1)(A)", "statute"),
             S("34 CFR § 99.31(a)(1)(i)(B)", "regulation")],
            answer="permitted under §1232g(b)(1)(A) and 34 CFR § 99.31(a)(1)(i)(B); "
                   "redisclosure barred (§ 99.33(a)).",
        )
    )
    check("not cfr-only", a.is_cfr_only is False)
    check("regulation dependent", a.regulation_dependent is True)
    check("usc cited in answer", a.usc_cited_in_answer is True)
    check("source cfr used", a.source_cfr_unused == [], str(a.source_cfr_unused))
    check("picks up non-source cfr (99.33)", "99.33" in a.cfr_sections_in_answer,
          str(a.cfr_sections_in_answer))


def test_redundant_unused_cfr() -> None:
    # Mirrors item 38: statutory definition used, CFR § 99.3 never cited.
    print("test_redundant_unused_cfr")
    a = analyze_item(
        rec(
            "38",
            [S("20 U.S.C. § 1232g(a)(4)(A)", "statute"),
             S("34 CFR § 99.3", "regulation")],
            answer="FERPA applies to education records maintained by the school; "
                   "no record existed at the time.",
        )
    )
    check("not regulation dependent", a.regulation_dependent is False)
    check("cfr source flagged unused", a.source_cfr_unused == ["34 CFR § 99.3"],
          str(a.source_cfr_unused))


def test_cfr_only_item() -> None:
    print("test_cfr_only_item")
    a = analyze_item(
        rec("36",
            [S("34 CFR § 99.31(a)(1)(i)(B)", "regulation"),
             S("34 CFR § 99.31(a)(1)(i)(A)", "regulation")],
            answer="qualifies under 34 CFR § 99.31(a)(1)(i)(B); not logged (§ 99.32(d)(1)).")
    )
    check("cfr-only true", a.is_cfr_only is True)


def test_guidance_plus_cfr_not_statute_fallback() -> None:
    # Item 43 shape: guidance + regulation, no statute.
    print("test_guidance_plus_cfr_not_statute_fallback")
    a = analyze_item(
        rec("43",
            [S("FPCO/PTAC Guidance ...", "guidance"),
             S("34 CFR § 99.5(a)(1)", "regulation")],
            answer="FERPA does not protect a deceased eligible student's records.")
    )
    check("cfr-only (no statute)", a.is_cfr_only is True)
    check("unused cfr source", a.source_cfr_unused == ["34 CFR § 99.5(a)(1)"],
          str(a.source_cfr_unused))


def test_summarize_buckets() -> None:
    print("test_summarize_buckets")
    records = [
        rec("1", [S("20 U.S.C. § 1232g(a)(1)(A)", "statute")]),
        rec("30", [S("20 U.S.C. § 1232g(b)(1)(A)", "statute"),
                   S("34 CFR § 99.31(a)(1)(i)(B)", "regulation")]),
        rec("36", [S("34 CFR § 99.31(a)(1)(i)(B)", "regulation")]),
        rec("43", [S("FPCO/PTAC Guidance ...", "guidance"),
                   S("34 CFR § 99.5(a)(1)", "regulation")]),
    ]
    s = summarize(records)
    check("total", s["total_items"] == 4, str(s))
    check("statute_only", s["statute_only"] == 1, str(s))
    check("with_regulation", s["with_regulation"] == 3, str(s))
    check("mixed", s["mixed_statute_and_regulation"] == 1, str(s))
    check("regulation_only", s["regulation_only"] == 1, str(s))
    check("reg_without_statute", s["regulation_without_statute_fallback"] == 2, str(s))
    check("with_guidance", s["with_guidance"] == 1, str(s))


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
