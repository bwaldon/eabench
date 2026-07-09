"""Analyze prevalence and necessity of regulation (C.F.R.) sources.

Addresses issue #7: (1) quantify how prevalent the USC/CFR mixture is, and
(2) surface, per item, the objective evidence needed to judge whether a CFR
provision is actually *required* to answer the item (vs. a redundant parallel
cite that could be deleted).

Necessity is ultimately a content/legal judgment; this script does NOT make
deletion decisions. It computes reproducible *signals* — is the item
regulation-only, does the expected answer actually invoke the CFR section,
how regulation-dependent is the answer's reasoning — that a human reviewer (see
``docs/cfr_necessity_findings.md``) uses to decide. Run: ``python analyze_sources.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark.jsonl"

# Citation scanners for *free text* (unanchored), tuned to the forms that
# appear in expected-answer prose: "34 CFR § 99.31(a)", bare "§ 99.33(a)",
# ranges "§§ 99.20-99.22", and bare statute refs "§1232g(b)(1)(A)".
_SUBDIVS = r"(?:\([^)\s]{1,10}\))*"
CFR_REF = re.compile(rf"(?:C\.?F\.?R\.?\s*)?§+\s*(?P<section>9\d\.\d+)(?P<subdivs>{_SUBDIVS})", re.IGNORECASE)
USC_REF = re.compile(rf"§+\s*(?P<section>1232g)(?P<subdivs>{_SUBDIVS})", re.IGNORECASE)

# Pull the CFR section number out of a canonical location like
# "34 CFR § 99.31(a)(1)(i)(B)".
_CFR_SECTION_IN_LOCATION = re.compile(r"§\s*(9\d\.\d+)")


@dataclass
class ItemAnalysis:
    task_id: str
    source_types: Counter
    cfr_locations: list[str]
    usc_locations: list[str]
    is_cfr_only: bool  # references CFR but has no statute source to fall back on
    cfr_sections_in_answer: set[str]  # every distinct CFR section cited in the answer
    usc_cited_in_answer: bool
    # per CFR source: is its section actually referenced in the expected answer?
    source_cfr_used: dict[str, bool] = field(default_factory=dict)

    @property
    def regulation_dependent(self) -> bool:
        """Answer's reasoning cites at least one CFR section."""
        return bool(self.cfr_sections_in_answer)

    @property
    def source_cfr_unused(self) -> list[str]:
        """CFR *source* locations whose section never appears in the answer."""
        return [loc for loc, used in self.source_cfr_used.items() if not used]

    def signal_verdict(self, cfr_location: str) -> str:
        """A deterministic verdict for one CFR source, from signals alone.

        Used only as a cheap prior to cross-check the LLM judge against — the
        LLM makes the real legal call. ESSENTIAL if the answer actually cites
        the section; REDUNDANT if it is unused but a statute could stand in;
        ESSENTIAL if there is no statute to fall back on; else REVIEW.
        """
        if self.source_cfr_used.get(cfr_location):
            return "ESSENTIAL"
        if self.usc_locations:
            return "REDUNDANT"
        if self.is_cfr_only:
            return "ESSENTIAL"
        return "REVIEW"


def _sections(text: str, pattern: re.Pattern) -> set[str]:
    return {m.group("section") for m in pattern.finditer(text)}


def analyze_item(record: dict) -> ItemAnalysis | None:
    sources = record.get("source", [])
    types = Counter(s["source_type"] for s in sources)
    cfr_locs = [s["location"] for s in sources if s["source_type"] == "regulation"]
    if not cfr_locs:
        return None  # only CFR-referencing items are in scope for issue #7
    usc_locs = [s["location"] for s in sources if s["source_type"] == "statute"]

    answer = record.get("expected_final_answer", "") or ""
    cfr_in_answer = _sections(answer, CFR_REF)
    usc_in_answer = bool(_sections(answer, USC_REF))

    used: dict[str, bool] = {}
    for loc in cfr_locs:
        m = _CFR_SECTION_IN_LOCATION.search(loc)
        section = m.group(1) if m else None
        used[loc] = bool(section and section in cfr_in_answer)

    return ItemAnalysis(
        task_id=record["task_id"],
        source_types=types,
        cfr_locations=cfr_locs,
        usc_locations=usc_locs,
        is_cfr_only=(types["statute"] == 0),
        cfr_sections_in_answer=cfr_in_answer,
        usc_cited_in_answer=usc_in_answer,
        source_cfr_used=used,
    )


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records: list[dict]) -> dict:
    total = len(records)
    with_cfr = with_guidance = usc_only = mixed = 0
    reg_only = reg_no_statute = 0
    for r in records:
        types = {s["source_type"] for s in r.get("source", [])}
        if "regulation" in types:
            with_cfr += 1
            if "statute" in types:
                mixed += 1
            else:
                reg_no_statute += 1  # nothing statutory to fall back on
                if types == {"regulation"}:
                    reg_only += 1
        if "guidance" in types:
            with_guidance += 1
        if types == {"statute"}:
            usc_only += 1
    return {
        "total_items": total,
        "statute_only": usc_only,
        "with_regulation": with_cfr,
        "mixed_statute_and_regulation": mixed,
        "regulation_only": reg_only,
        "regulation_without_statute_fallback": reg_no_statute,
        "with_guidance": with_guidance,
        "total_sources": sum(len(r.get("source", [])) for r in records),
        "sources_by_type": dict(
            Counter(s["source_type"] for r in records for s in r.get("source", []))
        ),
    }


def run_llm_judge(records, analyses, args):
    """Run the LLM necessity judge and return {task_id: JudgeResult} or None."""
    from necessity_judge import DEFAULT_CACHE, make_openai_responder, run_judge

    try:  # convenience: pick up OPENAI_API_KEY from a local .env if present
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    respond = make_openai_responder(args.model)
    return run_judge(
        records,
        respond=respond,
        model=args.model,
        cache_path=args.cache or DEFAULT_CACHE,
        refresh=args.refresh,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run the LLM necessity judge (needs OPENAI_API_KEY) and cross-check "
        "it against the deterministic signals.",
    )
    parser.add_argument("--model", default="gpt-5.4", help="LLM judge model (default: gpt-5.4).")
    parser.add_argument("--refresh", action="store_true", help="Ignore judge cache and re-query.")
    parser.add_argument("--cache", type=Path, default=None, help="Judge cache path.")
    args = parser.parse_args()

    records = load_records(args.benchmark)
    summary = summarize(records)
    analyses = [a for a in (analyze_item(r) for r in records) if a is not None]
    analyses.sort(key=lambda a: int(a.task_id))

    judged = run_llm_judge(records, analyses, args) if args.judge else None

    if args.json:
        payload = {
            "summary": summary,
            "items": [_item_payload(a, judged) for a in analyses],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print("PREVALENCE")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print(f"CFR-REFERENCING ITEMS ({len(analyses)})")
    print(
        f"  {'item':>4}  {'cfr-only':>8}  {'reg-dep':>7}  {'usc-in-ans':>10}  "
        f"cfr-sections-in-answer / unused-cfr-sources"
    )
    for a in analyses:
        unused = f"  UNUSED:{a.source_cfr_unused}" if a.source_cfr_unused else ""
        print(
            f"  {a.task_id:>4}  {str(a.is_cfr_only):>8}  "
            f"{str(a.regulation_dependent):>7}  {str(a.usc_cited_in_answer):>10}  "
            f"{sorted(a.cfr_sections_in_answer)}{unused}"
        )

    if judged is not None:
        _print_judge_section(analyses, judged)
    return 0


def _item_payload(a: "ItemAnalysis", judged) -> dict:
    payload = {
        "task_id": a.task_id,
        "cfr_locations": a.cfr_locations,
        "usc_locations": a.usc_locations,
        "is_cfr_only": a.is_cfr_only,
        "regulation_dependent": a.regulation_dependent,
        "usc_cited_in_answer": a.usc_cited_in_answer,
        "cfr_sections_in_answer": sorted(a.cfr_sections_in_answer),
        "source_cfr_unused": a.source_cfr_unused,
        "signal_verdicts": {loc: a.signal_verdict(loc) for loc in a.cfr_locations},
    }
    if judged is not None and a.task_id in judged:
        jr = judged[a.task_id]
        payload["llm_item_verdict"] = jr.item_verdict
        payload["llm_verdicts"] = jr.cfr_verdicts
        payload["disagreements"] = [
            {
                "location": loc,
                "signal": a.signal_verdict(loc),
                "llm": jr.cfr_verdicts.get(loc, {}).get("verdict"),
            }
            for loc in a.cfr_locations
            if jr.cfr_verdicts.get(loc, {}).get("verdict") != a.signal_verdict(loc)
        ]
        payload["model"] = jr.model
    return payload


def _print_judge_section(analyses, judged) -> None:
    print()
    print("LLM NECESSITY JUDGE (verdict | signal)")
    disagreements = []
    for a in analyses:
        jr = judged.get(a.task_id)
        if jr is None:
            continue
        print(f"  item {a.task_id}  [item_verdict: {jr.item_verdict}]")
        for loc in a.cfr_locations:
            signal = a.signal_verdict(loc)
            entry = jr.cfr_verdicts.get(loc, {})
            llm = entry.get("verdict", "?")
            flag = "" if llm == signal else "   <-- DISAGREE"
            if llm != signal:
                disagreements.append((a.task_id, loc, signal, llm))
            print(f"      {loc}")
            print(f"        LLM: {llm:<10} signal: {signal:<10}{flag}")
            if entry.get("rationale"):
                print(f"        rationale: {entry['rationale']}")
    print()
    if disagreements:
        print(f"DISAGREEMENTS ({len(disagreements)}) — review these:")
        for tid, loc, signal, llm in disagreements:
            print(f"  item {tid}: {loc}  signal={signal} llm={llm}")
    else:
        print("DISAGREEMENTS: none — LLM and signals agree on every CFR source.")


if __name__ == "__main__":
    raise SystemExit(main())
