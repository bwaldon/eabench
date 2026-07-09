"""Standalone tests for source_parser (run: ``python test_source_parser.py``).

No third-party test runner required; asserts + a tiny harness keep this
dependency-free. Cases are drawn from the real format variants in items 1-50.
"""

from __future__ import annotations

import sys

from source_parser import (
    DEFAULT_RELATIVE_BASE,
    leading_relative_path,
    parse_source_field,
    render_subdivs,
    validate_sources,
)

BASE = DEFAULT_RELATIVE_BASE.citation  # "20 U.S.C. § 1232g"

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} :: {detail}")
        _failures.append(name)


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r} want {want!r}")


# --- tokenizer -------------------------------------------------------------

def test_tokenizer() -> None:
    print("test_tokenizer")
    comps, _, rep = leading_relative_path("(a)(5)(A): text")
    eq("simple path", comps, ["a", "5", "A"])
    check("simple path not repaired", rep is False)

    comps, _, _ = leading_relative_path("(a) (5) (A)-(B): text")
    eq("spaced range", comps, ["a", "5", "A-B"])

    comps, _, rep = leading_relative_path("a)(4)(B)(i): text")
    eq("missing leading paren", comps, ["a", "4", "B", "i"])
    check("missing paren flagged repaired", rep is True)

    comps, _, _ = leading_relative_path("(b)(1)(H) No funds")
    eq("no colon", comps, ["b", "1", "H"])

    comps, _, _ = leading_relative_path("(a) (2)No funds shall")
    eq("space then runon text", comps, ["a", "2"])

    comps, _, _ = leading_relative_path("Nothing contained in this section")
    check("prose is not a path", comps is None)

    comps, _, _ = leading_relative_path("(d)Students' rather than parents'")
    eq("single subsection runon", comps, ["d"])


def test_render_subdivs() -> None:
    print("test_render_subdivs")
    eq("render simple", render_subdivs(["a", "5", "A"]), "(a)(5)(A)")
    eq("render range", render_subdivs(["a", "5", "A-B"]), "(a)(5)(A)-(B)")


# --- absolute citations ----------------------------------------------------

def test_absolute_usc() -> None:
    print("test_absolute_usc")
    src = parse_source_field(
        "20 U.S.C. § 1232g(b)(1)(E) No funds shall be made available"
    )
    eq("usc count", len(src), 1)
    eq("usc location", src[0]["location"], f"{BASE}(b)(1)(E)")
    eq("usc type", src[0]["source_type"], "statute")
    eq("usc text", src[0]["source_text"], "No funds shall be made available")


def test_absolute_usc_location_only_then_text() -> None:
    # Item 2: citation on one line, quoted text on the next.
    print("test_absolute_usc_location_only_then_text")
    src = parse_source_field(
        "20 U.S.C. § 1232g(b)(1)(A)\nNothing contained in this section"
    )
    eq("count", len(src), 1)
    eq("location", src[0]["location"], f"{BASE}(b)(1)(A)")
    eq("text joined from next line", src[0]["source_text"], "Nothing contained in this section")


def test_absolute_cfr() -> None:
    print("test_absolute_cfr")
    src = parse_source_field("34 CFR § 99.31(a)(1)(i)(B) A contractor, consultant")
    eq("cfr location", src[0]["location"], "34 CFR § 99.31(a)(1)(i)(B)")
    eq("cfr type", src[0]["source_type"], "regulation")

    # Item 38: section followed by a parenthetical *descriptor*, not a subdiv.
    src = parse_source_field('34 CFR § 99.3 (definition of "education records") The term')
    eq("cfr descriptor location", src[0]["location"], "34 CFR § 99.3")
    check(
        "cfr descriptor stays in text",
        src[0]["source_text"].startswith('(definition of "education records")'),
        src[0]["source_text"][:40],
    )


# --- relative paths --------------------------------------------------------

def test_relative_absolutized() -> None:
    print("test_relative_absolutized")
    src = parse_source_field("(a)(5)(A): For the purposes of this section")
    eq("relative location absolutized", src[0]["location"], f"{BASE}(a)(5)(A)")
    eq("relative type", src[0]["source_type"], "statute")
    eq("relative text", src[0]["source_text"], "For the purposes of this section")


def test_relative_range_and_spaces() -> None:
    print("test_relative_range_and_spaces")
    src = parse_source_field("(a) (5) (A)-(B): For the purposes")
    eq("range location", src[0]["location"], f"{BASE}(a)(5)(A)-(B)")


def test_relative_missing_paren() -> None:
    print("test_relative_missing_paren")
    src = parse_source_field("a)(4)(B)(i): (B)The term does not include")
    eq("repaired location", src[0]["location"], f"{BASE}(a)(4)(B)(i)")


# --- multi-provision splitting --------------------------------------------

def test_multi_relative() -> None:
    # Item 5: two distinct subsections.
    print("test_multi_relative")
    text = "(b)(1)(H) No funds ...\n(d) For the purposes of this section"
    src = parse_source_field(text)
    eq("two provisions", len(src), 2)
    eq("first loc", src[0]["location"], f"{BASE}(b)(1)(H)")
    eq("second loc", src[1]["location"], f"{BASE}(d)")


def test_mixed_statute_and_regulation() -> None:
    # Item 30: USC + CFR in one item.
    print("test_mixed_statute_and_regulation")
    text = (
        "20 U.S.C. § 1232g(b)(1)(A) Nothing in this section\n"
        "34 CFR § 99.31(a)(1)(i)(B) A contractor"
    )
    src = parse_source_field(text)
    eq("count", len(src), 2)
    eq("types", [s["source_type"] for s in src], ["statute", "regulation"])


def test_continuation_enumerator_not_split() -> None:
    # Item 1: a bare "(E):" line continues the previous provision's text and
    # must NOT become a bogus "...1232g(E)" citation.
    print("test_continuation_enumerator_not_split")
    text = (
        "20 U.S.C. § 1232g(b)(1)(E) No funds\n"
        "(E): State and local officials\n"
        "20 U.S.C. § 1232g(b)(1)(J) Nothing contained"
    )
    src = parse_source_field(text)
    eq("two real provisions", len(src), 2)
    eq("first loc", src[0]["location"], f"{BASE}(b)(1)(E)")
    check(
        "bare (E) folded into first text",
        "State and local officials" in src[0]["source_text"],
        src[0]["source_text"],
    )
    eq("second loc", src[1]["location"], f"{BASE}(b)(1)(J)")


def test_roman_child_not_split() -> None:
    # Item 19: (i)/(ii) under a deep path are continuations, not subsections.
    print("test_roman_child_not_split")
    text = (
        "a)(4)(B)(i): (B)The term does not include—\n"
        "(i)records of instructional personnel\n"
        "(ii)records maintained by a law enforcement unit"
    )
    src = parse_source_field(text)
    eq("single provision", len(src), 1)
    eq("location", src[0]["location"], f"{BASE}(a)(4)(B)(i)")
    check(
        "roman children folded in",
        "(ii)records maintained" in src[0]["source_text"],
        src[0]["source_text"][-60:],
    )


def test_subsection_i_is_new_when_shallow() -> None:
    # Item 16: leading "(i)" is subsection (i), a new provision.
    print("test_subsection_i_is_new_when_shallow")
    src = parse_source_field("(i)Drug and alcohol violation disclosures: (1)In general")
    eq("count", len(src), 1)
    eq("location", src[0]["location"], f"{BASE}(i)")


# --- guidance / mixtures ---------------------------------------------------

def test_guidance() -> None:
    # Item 43: agency guidance line + a CFR line.
    print("test_guidance")
    text = (
        "FPCO/PTAC Guidance (informed by 20 U.S.C. § 1232g(d) and common law "
        "principles): Once an eligible student dies, FERPA rights lapse.\n"
        "34 CFR § 99.5(a)(1) When a student becomes an eligible student"
    )
    src = parse_source_field(text)
    eq("count", len(src), 2)
    eq("guidance type", src[0]["source_type"], "guidance")
    check(
        "guidance location is the descriptor",
        src[0]["location"].startswith("FPCO/PTAC Guidance"),
        src[0]["location"],
    )
    check(
        "guidance text after colon",
        src[0]["source_text"].startswith("Once an eligible student dies"),
        src[0]["source_text"][:40],
    )
    eq("second is regulation", src[1]["source_type"], "regulation")


# --- misc ------------------------------------------------------------------

def test_empty() -> None:
    print("test_empty")
    eq("empty string", parse_source_field(""), [])
    eq("whitespace", parse_source_field("   \n  "), [])


def test_raw_preserved() -> None:
    print("test_raw_preserved")
    src = parse_source_field("(a)(5)(A): For the purposes")
    eq("raw kept", src[0]["raw"], "(a)(5)(A): For the purposes")


def test_validate_flags_guidance_and_empty() -> None:
    print("test_validate_flags_guidance_and_empty")
    warns = validate_sources([])
    check("empty list warns", warns == ["no source parsed"], str(warns))
    src = parse_source_field("(a)(5)(A): For the purposes")
    check("clean source no warnings", validate_sources(src) == [], str(validate_sources(src)))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print()
    if _failures:
        print(f"FAILED {len(_failures)} check(s): {_failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
