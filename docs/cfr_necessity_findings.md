# CFR-source prevalence & necessity — findings

Status: **edits applied after maintainer review — 2 redundant CFR passages removed (see §6)** ·
Issue: [#7](https://github.com/bwaldon/eabench/issues/7) ·
Reproduce: `python analyze_sources.py` (signals) · `python analyze_sources.py --judge` (adds the LLM judge)

Issue #7 asks two things: (1) how prevalent is the USC/CFR mixture, and
(2) whether items that reference CFR actually *require* the CFR provision to be
answerable — if not, the CFR passage should be deleted. This document answers
(1) with hard counts and (2) with a reproducible, two-signal judgment per CFR
source. Depends on the structured `source` schema from #3.

## 1. Prevalence

Of **50 items** / **84 sources** (69 statute, 14 regulation, 1 guidance):

| bucket | count | items |
|---|---|---|
| statute-only | 40 | — |
| references any regulation (CFR) | **10** | 30, 36, 37, 38, 39, 41, 43, 45, 47, 50 |
| — mixed statute **and** regulation | 7 | 30, 37, 38, 39, 41, 47, 50 |
| — regulation-only (no other authority) | 2 | 36, 45 |
| — regulation + guidance (no statute) | 1 | 43 |
| references guidance | 1 | 43 |

The mixture affects **20% of the benchmark** and is concentrated, not pervasive.

> **Post-edit state (after §6).** Removing the two redundant CFR passages leaves
> **9** CFR-referencing items and **82** sources (69 statute, **12** regulation,
> 1 guidance): 6 mixed, 2 regulation-only, 1 regulation+guidance. The tables
> below are stated as of *analysis time* (before the edits).

## 2. How necessity is judged (two independent signals)

Necessity is a legal judgment, so it is assessed two ways and the two are
cross-checked. **Neither modifies any item.**

1. **Deterministic signal** (`analyze_sources.py`, no LLM). A cheap prior per
   CFR source: ESSENTIAL if the expected answer actually cites that section;
   REDUNDANT if it is uncited but a statute source could stand in; ESSENTIAL if
   there is no statute to fall back on; else REVIEW. Fully reproducible.
2. **LLM judge** (`necessity_judge.py`, `--judge`). An OpenAI model
   (`gpt-5.4`, matching `eval_pipeline.py`) is given the query, **all** source
   texts, and the expected answer, and classifies each CFR source
   ESSENTIAL / REDUNDANT / REVIEW with a rationale. Strict-JSON output,
   validated; verdicts cached to `necessity_judge_results.json` (keyed by an
   input+rubric+model fingerprint) so re-runs are free and the doc's provenance
   is captured. The LLM makes the real legal call; the signal is the check.

Where the two **disagree**, the item is queued for human review (§4). The key
distinction the LLM draws that the signal cannot: an answer *mentioning* a
regulation is not the same as the regulation being *decisive* — several answers
cite a CFR section whose rule is actually supplied by the statute.

## 3. Results (`gpt-5.4`, cross-checked)

| item | CFR source | LLM | signal | agree | note |
|---|---|---|---|---|---|
| 30 | § 99.31(a)(1)(i)(B) | ESSENTIAL | ESSENTIAL | ✓ | adds direct-control test central to the facts |
| 36 | § 99.31(a)(1)(i)(B) | ESSENTIAL | ESSENTIAL | ✓ | regulation-only; outsourced-official rule |
| 36 | § 99.31(a)(1)(i)(A) | ESSENTIAL | ESSENTIAL | ✓ | "legitimate educational interest" standard |
| 37 | § 99.12(a) | ESSENTIAL | ESSENTIAL | ✓ | multi-student redaction rule decides the case |
| 38 | § 99.3 | **REDUNDANT** | **REDUNDANT** | ✓ | duplicates statutory "education records" definition |
| 39 | § 99.7(a)(1) | ESSENTIAL | ESSENTIAL | ✓ | annual-timing requirement |
| 39 | § 99.7(b) | ESSENTIAL | ESSENTIAL | ✓ | "any means reasonably likely to inform" — decisive |
| 41 | § 99.21(a) | REDUNDANT | ESSENTIAL | ✗ | LLM: hearing right restates (a)(2); answer's core follows from statute |
| 43 | § 99.5(a)(1) | REDUNDANT | ESSENTIAL | ✗ | LLM: answer turns on death-lapse guidance, not this reg |
| 45 | § 99.31(a)(1)(i)(B) | ESSENTIAL | ESSENTIAL | ✓ | regulation-only; school-official conditions |
| 45 | § 99.33(a)(1) | ESSENTIAL | ESSENTIAL | ✓ | no-redisclosure limitation incorporated by the exception |
| 47 | § 99.31(a)(9)(ii)(A) | ESSENTIAL | ESSENTIAL | ✓ | grand-jury / non-disclosure exception |
| 50 | § 99.5(a)(1) | REDUNDANT | ESSENTIAL | ✗ | LLM: restates § 1232g(d) rights-transfer |
| 50 | § 99.5(b) | REDUNDANT | ESSENTIAL | ✗ | LLM: decisive dependent-student rule is elsewhere |

**Agreement:** 10 of 14 CFR sources agree. Both signals call **item 38's § 99.3
REDUNDANT** — the one consensus deletion. Six items are unanimously ESSENTIAL
(30, 36, 37, 39, 45, 47).

## 4. Human-review queue — the 4 disagreements

These are the cases where the LLM judges a regulation REDUNDANT even though the
expected answer cites it (hence the signal called it ESSENTIAL). Each was given
a maintainer's legal call — **see §6 for the resolution** (41, 43 kept; 50's
§ 99.5(a)(1) removed; 38 removed by consensus):

- **41 · § 99.21(a).** The answer requires a hearing and a written statement,
  but cites §§ 99.22 / 99.21(b)(2) for those — not the listed § 99.21(a). The
  amendment-scope holding follows from § 1232g(a)(2). *Plausibly the cited
  subsection should change rather than be deleted.*
- **43 · § 99.5(a)(1).** Governs rights-transfer when a student *becomes*
  eligible; the answer is about a *deceased* eligible student and rests on
  FPCO/PTAC guidance. The regulation is tangential. (Item has no statute source,
  so removing it leaves guidance-only grounding — itself worth flagging.)
- **50 · § 99.5(a)(1) and § 99.5(b).** § 99.5(a)(1) restates § 1232g(d). The
  answer name-drops § 99.5(b) for the dual-enrollment information-exchange
  point, but the *decisive* dependent-student rule is § 1232g(b)(1)(H) /
  § 99.31(a)(8) — neither of which is in the item's source list.

## 5. Bottom line

The USC/CFR mixture is **mostly load-bearing, not accidental.** By the stricter
LLM standard, 6 of 10 items are unambiguously regulation-dependent (often on
regulation-only detail with no statutory equivalent — §§ 99.7(b), 99.12(a),
99.5(b), 99.31 conditions). Removal is warranted for **item 38** (consensus) and
should be *considered* for the three disagreement items (41, 43, 50) pending a
human legal call. The LLM's recurring insight: in items 41/50, the answer
*cites* the regulation but the *decisive* authority is statutory — so those
CFR passages may be prunable even though the text mentions them. On close
inspection this held only partially: item 50's § 99.5(a)(1) was prunable but
its § 99.5(b) was load-bearing, and item 41's CFR proved load-bearing after
all (§6).

## 6. Actions taken (maintainer-reviewed)

Each flagged source was adjudicated against the actual item text (does the
expected answer *rely* on the regulation, and would deletion leave a dangling
citation?). Two passages were removed; two were kept.

| item | action | reason |
|---|---|---|
| **38** | **removed** `34 CFR § 99.3` | Redundant with the statutory "education records" definition (§ 1232g(a)(4)(A)); the answer never cites it. All three checks agree. |
| **50** | **removed** `34 CFR § 99.5(a)(1)`; **kept** `34 CFR § 99.5(b)` | § 99.5(a)(1) merely restates § 1232g(d) and is uncited. § 99.5(b) is the operative dual-enrollment carve-out, cited in both the answer and the Concepts. With § 99.5(a)(1) gone, the judge re-rates § 99.5(b) **ESSENTIAL**. |
| **41** | **kept** `34 CFR § 99.21(a)` | Despite the REDUNDANT verdict, the answer's decisive procedural obligations (hearing under **§ 99.22**, written statement under **§ 99.21(b)(2)**) are regulation-only; deleting the CFR would strip grounding the answer depends on. |
| **43** | **kept** `34 CFR § 99.5(a)(1)` | It is the item's only *codified* authority (the other source is FPCO/PTAC guidance) and supports the "eligible student" predicate the follow-up turns on. Removal would leave the benchmark's sole guidance-only item. |

Net effect: `regulation` sources 14 → 12; CFR-referencing items 10 → 9; remaining
signal/judge disagreements are items 41 and 43, both **deliberately kept** for the
reasons above.

The edits were made to `items/38/full_item.txt` and `items/50/full_item.txt`
(the source of truth), and `benchmark.jsonl` was regenerated. Reproduce/verify:
`python build_benchmark.py --check && python analyze_sources.py --judge`.
