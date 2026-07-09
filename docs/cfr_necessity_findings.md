# CFR-source prevalence & necessity — findings

Status: **analysis / recommendations only — no benchmark items were modified** ·
Issue: [#7](https://github.com/bwaldon/eabench/issues/7) ·
Reproduce: `python analyze_sources.py` (add `--json` for machine-readable output)

Issue #7 asks two things: (1) how prevalent is the USC/CFR mixture, and
(2) whether items that reference CFR actually *require* the CFR provision to be
answerable — if not, the CFR passage should be deleted. This document answers
(1) with hard counts and (2) with a per-item, evidence-backed assessment.
Depends on the structured `source` schema from #3 (`source_type` distinguishes
statute / regulation / guidance).

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

So the mixture affects **20% of the benchmark** and is concentrated, not
pervasive.

## 2. How necessity was assessed

Necessity is a legal/content judgment, so this is **not** fully automated.
`analyze_sources.py` computes reproducible signals; the verdict column below is
a human judgment layered on top of them. The signals:

- **cfr-only** — the item has no statute source, so the CFR *cannot* be deleted
  without leaving the item ungrounded.
- **source CFR used?** — does the expected answer actually cite the section of
  each CFR *source*? An unused CFR source is a redundancy signal.
- **regulation-dependent?** — does the answer's reasoning cite *any* CFR section
  (including ones beyond the listed sources, e.g. § 99.32/§ 99.33)? This shows
  whether the model must reach regulatory detail to answer.

Signal table (from `analyze_sources.py`):

| item | cfr-only | reg-dependent | CFR sections cited in answer | unused CFR source |
|---|---|---|---|---|
| 30 | no  | yes | 99.31, 99.33 | — |
| 36 | yes | yes | 99.31, 99.32, 99.33 | — |
| 37 | no  | yes | 99.12 | — |
| 38 | no  | **no** | — | **§ 99.3** |
| 39 | no  | yes | 99.7 | — |
| 41 | no  | yes | 99.20, 99.21, 99.22 | — |
| 43 | yes | **no** | — | **§ 99.5(a)(1)** |
| 45 | yes | yes | 99.30, 99.31 | § 99.33(a)(1) |
| 47 | no  | yes | 99.31, 99.32 | — |
| 50 | no  | yes | 99.31, 99.5 | — |

## 3. Per-item verdicts

**Legend:** ESSENTIAL = CFR supplies the operative rule the answer turns on;
REDUNDANT = CFR restates authority the item's statute already provides and the
answer does not use it; REVIEW = present but not clearly required, needs a human
call.

### ESSENTIAL — keep the CFR (8 items)

- **30** — Statute (b)(1)(A) grants the school-official exception in the
  abstract; the decisive **four-part operational test** (designated in annual
  notice, direct control, redisclosure bar, institutional function) lives in
  § 99.31(a)(1)(i)(B). The answer walks all four prongs and additionally invokes
  § 99.33(a). Statute alone is insufficient.
- **36** — Regulation-only; the entire school-official / threat-assessment-team
  analysis and the § 99.32(d)(1) logging carve-out are regulatory. Nothing to
  fall back on.
- **37** — Statute (a)(1)(A) gives the inspection right; the **multi-student
  redaction/segregation rule** that decides the case is § 99.12(a). The answer
  is built on it.
- **39** — Statute (e) requires annual notice; the dispositive
  "**by any means reasonably likely to inform**" standard is § 99.7(b). The
  answer hinges on it.
- **41** — Statute (a)(2) gives the amendment right; the **hearing and
  written-statement procedure** (§§ 99.20–99.22) that the answer requires the
  school to follow is regulatory.
- **45** — Regulation-only; § 99.31(a)(1)(i)(B) is the whole basis of the
  answer. (Note: the second source, § 99.33(a)(1), is only supporting — the
  answer references redisclosure conceptually and cites § 99.30(b) rather than
  § 99.33 — but at least one CFR source is indispensable.)
- **47** — Statutes (b)(1)(J)(i)/(b)(2)(B) cover the subpoena; the
  **no-notice** rule § 99.31(a)(9)(ii)(A) and the **logging exemption**
  § 99.32(d)(5) are regulatory-only and both appear in the answer.
- **50** — Statute (d) gives rights-transfer; § 99.5(b)'s **dual-enrollment
  information-exchange** rule is regulatory-only and is expressly invoked in the
  answer.

### REDUNDANT — CFR removal candidate (1 item)

- **38** — Both sources define "education records": statute **§ 1232g(a)(4)(A)**
  and regulation **§ 99.3**. The answer reasons entirely from the statutory
  definition ("information … not yet contained in any education record
  maintained by the school") and **never cites § 99.3** (regulation-dependent:
  no). The item is fully answerable from the statute.
  **Recommendation:** delete the `34 CFR § 99.3` source from item 38. *(Caveat:
  § 99.3 is the more commonly quoted working definition and adds itemized
  exclusions; if the maintainers want the regulatory definition on record, keep
  it — but it is not load-bearing for this item's answer.)*

### REVIEW — not clearly required (1 item)

- **43** — Guidance + § 99.5(a)(1). The answer is driven by FPCO/PTAC guidance
  on **deceased** eligible students; § 99.5(a)(1) governs rights-*transfer* when
  a student *becomes* eligible, a background predicate the answer **never
  cites** (regulation-dependent: no). It is a tangential authority rather than
  the governing rule.
  **Recommendation:** review. Deleting § 99.5(a)(1) would leave the item grounded
  only in guidance (it has no statute source) — which is itself worth flagging,
  since this is the benchmark's only guidance-grounded item. The deeper question
  is whether item 43 should cite the operative authority (§ 1232g(d) + guidance)
  rather than § 99.5(a)(1).

## 4. Bottom line

The USC/CFR mixture is **mostly load-bearing, not accidental**: in 8 of 10
CFR-referencing items the regulation supplies the operative rule the answer
depends on (frequently regulation-only detail such as §§ 99.7(b), 99.12(a),
99.5(b), 99.32(d)(5) that has no statutory equivalent). Only **item 38** has a
clearly redundant CFR passage safe to delete, and **item 43** warrants a
targeted review. Per the agreed scope, **no items were modified**; §5 lists the
concrete follow-up edits for maintainer approval.

## 5. Proposed follow-up (requires approval — not applied here)

1. **Item 38:** remove `34 CFR § 99.3` (redundant with statutory definition).
2. **Item 43:** review the source set; consider replacing/removing
   `34 CFR § 99.5(a)(1)` in favor of the operative authority, and reconsider
   guidance-only grounding.
3. Re-run `python build_benchmark.py --check` and `python analyze_sources.py`
   after any edit to confirm the counts move as expected.
