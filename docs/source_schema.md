# Benchmark `source` field — specification

Status: **active** · Supersedes: the free-text `statutory_provision` field ·
Issue: [#3](https://github.com/bwaldon/eabench/issues/3)

## 1. Motivation

Each benchmark item is grounded in one or more legal provisions. Historically
these were stored in a single free-text field, `statutory_provision`, whose
format drifted item-to-item:

- Some items cited **absolute** locations — `20 U.S.C. § 1232g(b)(1)(G) …`.
- Others used **relative** paths within FERPA — `(a)(5)(A): …`.
- Items that depend on **several** provisions crammed them into one blob.
- Regulatory (`34 CFR …`) and statutory (`20 U.S.C. …`) authorities were mixed
  with no way to tell them apart programmatically.

This blocks two things the benchmark needs going forward: (a) it will not stay
FERPA-specific, so provisions must be identified by absolute citation; and
(b) retrieval-style evals need the citation **location** cleanly separated from
the quoted **source text**.

This spec defines a structured replacement, `source`, and the parser that
produces it (`source_parser.py`).

## 2. Schema

`source` is a JSON **array** (an item may cite multiple provisions). Each
element is an object:

| field         | type   | required | description |
|---------------|--------|----------|-------------|
| `location`    | string | yes      | Absolute, canonical citation of the provision (see §3). For `guidance`, a human-readable descriptor. |
| `source_text` | string | yes      | The quoted text of the law/regulation/guidance, with the leading citation stripped. |
| `source_type` | enum   | yes      | One of `statute`, `regulation`, `guidance` (see §4). |
| `raw`         | string | yes      | The verbatim original segment (citation + text), preserved for auditing and re-parsing. Not intended for eval scoring. |

### Rationale for the split

- **`location` vs `source_text`.** Retrieval evals score whether a model
  surfaced the right *provision* (`location`); answer-quality evals use the
  *text*. Keeping them separate lets each eval use exactly what it needs.
- **`source` as a list.** A single query can turn on multiple provisions (e.g.
  a statute plus its implementing regulation). Nesting distinct provisions as
  list elements — rather than concatenating — makes per-provision scoring and
  counting possible.
- **`raw`.** Guarantees no information is lost in normalization and lets the
  pipeline be re-run if the parser improves.

### Example

Item 30 (a statute + its implementing regulation):

```json
"source": [
  {
    "location": "20 U.S.C. § 1232g(b)(1)(A)",
    "source_text": "Nothing in this section shall be construed to prohibit … disclosing education records … to other school officials …",
    "source_type": "statute",
    "raw": "20 U.S.C. § 1232g(b)(1)(A) Nothing in this section …"
  },
  {
    "location": "34 CFR § 99.31(a)(1)(i)(B)",
    "source_text": "A contractor, consultant, volunteer, or other party to whom an agency or institution has outsourced …",
    "source_type": "regulation",
    "raw": "34 CFR § 99.31(a)(1)(i)(B) A contractor …"
  }
]
```

## 3. `location` canonical form

Absolute citations are the target for every `statute` and `regulation` source.

- **Statute (U.S. Code):** `<title> U.S.C. § <section><subdivisions>`
  e.g. `20 U.S.C. § 1232g(b)(1)(J)(i)`
- **Regulation (C.F.R.):** `<title> CFR § <section><subdivisions>`
  e.g. `34 CFR § 99.31(a)(1)(i)(B)`

Rules:

- `§` is used (not "Sec." / "section"); single spaces around it.
- Subdivisions are contiguous parenthesized components with no internal spaces:
  `(a)(5)(A)`. Ranges are written `(A)-(B)`.
- **Relative paths are absolutized.** A bare FERPA path such as `(a)(5)(A)` is
  expanded by prepending the base citation for the statute the benchmark
  currently covers, `20 U.S.C. § 1232g`, yielding `20 U.S.C. § 1232g(a)(5)(A)`.
  The base is declared once, as `DEFAULT_RELATIVE_BASE` in `source_parser.py`.

> **Alignment with P4P.** The `location` format is intended to match how
> provisions are keyed in P4P ingestion. That representation was not available
> when this spec was written; the Bluebook-style form above is the working
> standard. If P4P keys provisions differently, change the scheme `template`s
> (and, if needed, `DEFAULT_RELATIVE_BASE`) in `source_parser.py` — the rest of
> the pipeline is agnostic to the exact string.

## 4. `source_type`

| value        | when |
|--------------|------|
| `statute`    | U.S. Code citations (and absolutized FERPA-relative paths). |
| `regulation` | Code of Federal Regulations citations. |
| `guidance`   | Agency guidance / interpretive material that is not itself a statute or regulation (e.g. FPCO/PTAC guidance), possibly informed by one. |

The set is data-driven (`SOURCE_TYPES` in `source_parser.py`); adding a new
authority scheme extends it automatically.

## 5. Parsing behavior (`source_parser.py`)

`parse_source_field(text, *, relative_base=DEFAULT_RELATIVE_BASE) -> list[dict]`
turns a raw provision blob into the `source` array. It is intentionally
**configuration-driven** — citation authorities (`CITATION_SCHEMES`) and the
relative base are declared as data, not hardcoded into the control flow, so the
benchmark can add statutes/regulations without rewriting the parser.

### Segmentation (splitting into multiple provisions)

The blob is split line-by-line. A line **starts a new provision** when it opens
with:

- an absolute citation (`20 U.S.C. …` / `34 CFR …`), **or**
- a relative path with ≥2 components (`(a)(5)(A)`), **or**
- a single **lowercase-letter** subsection (`(d)`, `(e)`).

A line that opens with a single bare enumerator (an uppercase letter, a digit,
or a multi-letter roman numeral such as `(E)`, `(1)`, `(ii)`) is treated as a
**continuation** of the current provision's text — never promoted to its own
citation, which would fabricate a wrong location like `…1232g(E)`. The
`(i)` case (subsection *i* vs. roman *one*) is disambiguated by context: inside
a deep path it is a roman child (continuation); otherwise it is subsection `(i)`.

Lines with no recognizable citation are continuation text of the current
provision, unless they are the first line, in which case they form a
`guidance` source.

### Normalization handled

The parser tolerates and canonicalizes the real-world variants found in the
items, including: missing leading parenthesis (`a)(4)(B)(i)`), internal spaces
(`(a) (5) (A)`), ranges (`(A)-(B)`), optional trailing colon, text on the same
line as the citation or on the following line, and C.F.R. sections followed by
a parenthetical descriptor rather than a subdivision (`34 CFR § 99.3
(definition …)`).

## 6. Validation

`validate_sources(sources) -> list[str]` returns warnings for low-confidence
output: empty `source_text`/`location`, unknown `source_type`, or a
non-`guidance` location that is not canonical. `build_benchmark.py` runs this
for every item; `python build_benchmark.py --check` exits non-zero if any item
has missing fields or source warnings. As of this spec, all 50 items parse with
zero warnings (69 statute, 14 regulation, 1 guidance sources).

## 7. Regenerating

```bash
python build_benchmark.py            # regenerate benchmark.jsonl
python build_benchmark.py --check    # regenerate + fail on any warning
python test_source_parser.py         # unit tests for the parser
```

## 8. Downstream compatibility

`eval_pipeline.py` and `analyze_results.py` consume `task_id`, `user_query`,
and `gold_exchange` — none of which changed. Any future eval that scores
retrieval should read `source[].location`; one that scores answer grounding
should read `source[].source_text`.
