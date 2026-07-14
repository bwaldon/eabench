"""Parse free-text legal-provision fields into a standardized ``source`` list.

Benchmark items historically stored the governing law in a single
``statutory_provision`` string whose format varied item-to-item:

* absolute U.S. Code citations   -- ``20 U.S.C. § 1232g(b)(1)(E) No funds ...``
* absolute C.F.R. citations      -- ``34 CFR § 99.31(a)(1)(i)(B) A contractor ...``
* FERPA-relative paths           -- ``(a)(5)(A): For the purposes ...``
* several malformed variants     -- missing leading paren (``a)(4)(B)(i):``),
                                    internal spaces (``(a) (5) (A)-(B):``),
                                    ranges, colons-or-not after the path
* agency guidance / mixtures      -- ``FPCO/PTAC Guidance (informed by ...): ...``

This module turns any of those into a list of structured records

    {"location": str, "source_text": str, "source_type": str, "raw": str}

so that (a) a single item can cite several provisions, (b) provisions are
identified by *absolute* citations regardless of how they were written, and
(c) ``location`` (used for retrieval evals) is cleanly separated from
``source_text`` (the quoted law). See ``docs/source_schema.md`` for the spec.

The parser is deliberately configuration-driven: citation schemes and the base
citation used to absolutize relative paths are declared as data
(:data:`CITATION_SCHEMES`, :data:`DEFAULT_RELATIVE_BASE`) rather than baked into
the parsing logic, so the benchmark can grow beyond FERPA without edits here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration (the only place citation authorities and defaults are declared)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationScheme:
    """A recognizable *absolute* citation authority.

    ``regex`` must expose three named groups when it matches the start of a
    line: ``title`` (e.g. ``20``), ``section`` (e.g. ``1232g`` / ``99.31``) and
    ``subdivs`` (the contiguous ``(a)(1)(i)`` run immediately following the
    section, possibly empty). ``template`` renders the canonical form.
    """

    name: str
    source_type: str
    regex: re.Pattern
    template: str

    def canonical(self, match: re.Match) -> str:
        return self.template.format(
            title=match.group("title"),
            section=match.group("section"),
            subdivs=match.group("subdivs") or "",
        )


@dataclass(frozen=True)
class RelativeBase:
    """Citation prepended to bare, source-relative subdivision paths.

    FERPA is codified at ``20 U.S.C. § 1232g``, so a relative path like
    ``(a)(5)(A)`` denotes ``20 U.S.C. § 1232g(a)(5)(A)``. When the benchmark
    adds items governed by a different statute, pass a different
    :class:`RelativeBase` to :func:`parse_source_field` (or, preferably, author
    those items with absolute citations).
    """

    citation: str
    source_type: str


# Absolute citation authorities. Ordered most-specific first is unnecessary
# here because the leading title/authority tokens are mutually exclusive.
CITATION_SCHEMES: list[CitationScheme] = [
    CitationScheme(
        name="usc",
        source_type="statute",
        # 20  U.S.C.  §  1232g (b)(1)(E)     -- dots/spaces/§ all optional.
        regex=re.compile(
            r"^\s*(?P<title>\d+)\s*U\.?\s*S\.?\s*C\.?\s*§?\s*"
            r"(?P<section>\d+[A-Za-z]*(?:-\d+)?)"
            r"(?P<subdivs>(?:\([^)]{1,15}\))*)",
            re.IGNORECASE,
        ),
        template="{title} U.S.C. § {section}{subdivs}",
    ),
    CitationScheme(
        name="cfr",
        source_type="regulation",
        # 34  C.F.R.  §  99.31 (a)(1)(i)(B)
        regex=re.compile(
            r"^\s*(?P<title>\d+)\s*C\.?\s*F\.?\s*R\.?\s*§?\s*"
            r"(?P<section>\d+(?:\.\d+)*)"
            r"(?P<subdivs>(?:\([^)]{1,15}\))*)",
            re.IGNORECASE,
        ),
        template="{title} CFR § {section}{subdivs}",
    ),
]

# FERPA statutory base for relative subdivision paths.
DEFAULT_RELATIVE_BASE = RelativeBase(citation="20 U.S.C. § 1232g", source_type="statute")

# Source-type given to leading text that matches no citation scheme.
GUIDANCE_SOURCE_TYPE = "guidance"

# The set of values ``source_type`` can take, for validators/consumers.
SOURCE_TYPES = frozenset(
    {s.source_type for s in CITATION_SCHEMES}
    | {DEFAULT_RELATIVE_BASE.source_type, GUIDANCE_SOURCE_TYPE}
)

# ---------------------------------------------------------------------------
# Relative-path tokenizer
# ---------------------------------------------------------------------------

# One subdivision component: an optional opening paren (so a dropped leading
# paren such as "a)(4)(B)" is recoverable), a short alnum token, a close paren.
_COMP_RE = re.compile(r"\s*(?P<open>\()?\s*(?P<comp>[A-Za-z]{1,5}|\d{1,3})\s*\)")
# A range continuation: "-(B)" following a component, e.g. "(A)-(B)".
_RANGE_RE = re.compile(r"\s*-\s*\(?\s*(?P<comp>[A-Za-z]{1,5}|\d{1,3})\s*\)")


def leading_relative_path(line: str) -> tuple[list[str] | None, int, bool]:
    """Consume a leading run of subdivision components from ``line``.

    Returns ``(components, chars_consumed, repaired)`` where ``components`` is
    e.g. ``["a", "5", "A-B"]`` (ranges kept as ``"A-B"``), or ``(None, 0,
    False)`` when the line does not begin with a subdivision path. ``repaired``
    is True when the first component's opening parenthesis was missing.
    """
    comps: list[str] = []
    pos = 0
    first = True
    repaired = False
    while True:
        m = _COMP_RE.match(line, pos)
        if not m:
            break
        has_open = m.group("open") is not None
        comp = m.group("comp")
        if not has_open:
            # Only the *first* token may be missing its opening paren, and only
            # if it is a short id (avoids swallowing prose like "Records)").
            if not first or len(comp) > 2:
                break
            repaired = True
        end = m.end()
        rng = _RANGE_RE.match(line, end)
        if rng:
            comp = f"{comp}-{rng.group('comp')}"
            end = rng.end()
        comps.append(comp)
        pos = end
        first = False
    if not comps:
        return None, 0, False
    return comps, pos, repaired


def render_subdivs(comps: list[str]) -> str:
    """Render tokenized components back to ``(a)(5)(A)-(B)`` form."""
    parts = []
    for c in comps:
        if "-" in c:
            lo, hi = c.split("-", 1)
            parts.append(f"({lo})-({hi})")
        else:
            parts.append(f"({c})")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Opening-line classification
# ---------------------------------------------------------------------------


@dataclass
class Opening:
    """How a provision-opening line was recognized."""

    kind: str  # "absolute" | "relative"
    source_type: str
    location: str | None  # set for absolute; built later for relative
    remainder: str  # text on the same line after the citation/path
    depth: int  # number of subdivision levels (for continuation heuristics)
    repaired: bool
    comps: list[str] | None


def classify_opening(line: str) -> Opening | None:
    """Classify a line's leading citation, or ``None`` if it is prose."""
    for scheme in CITATION_SCHEMES:
        m = scheme.regex.match(line)
        if m:
            subdivs = m.group("subdivs") or ""
            return Opening(
                kind="absolute",
                source_type=scheme.source_type,
                location=scheme.canonical(m),
                remainder=line[m.end() :],
                depth=subdivs.count("("),
                repaired=False,
                comps=None,
            )
    comps, consumed, repaired = leading_relative_path(line)
    if comps:
        return Opening(
            kind="relative",
            source_type="",  # filled from the RelativeBase at build time
            location=None,
            remainder=line[consumed:],
            depth=len(comps),
            repaired=repaired,
            comps=comps,
        )
    return None


_SINGLE_LOWER_RE = re.compile(r"[a-z]")


def opens_new_provision(opening: Opening | None, current: Opening | None) -> bool:
    """Decide whether ``opening`` starts a new provision vs. continues ``current``.

    Absolute citations and multi-component relative paths always start a new
    provision. A single bare component is treated as a *continuation
    enumerator* (e.g. a trailing ``(E)`` or ``(1)`` belonging to the previous
    provision's text) unless it is a single lowercase letter, which denotes a
    top-level subsection. The roman-numeral-vs-subsection ambiguity of ``(i)``
    is resolved by context: inside a deep path it is a roman child
    (continuation), otherwise a subsection (new).
    """
    if opening is None:
        return False
    if opening.kind == "absolute":
        return True
    comps = opening.comps or []
    if len(comps) >= 2:
        return True
    comp = comps[0]
    if _SINGLE_LOWER_RE.fullmatch(comp):
        if comp == "i":
            current_depth = current.depth if current is not None else 0
            return current_depth < 3
        return True
    return False


# ---------------------------------------------------------------------------
# Text cleanup and record building
# ---------------------------------------------------------------------------

_LEADING_COLON_RE = re.compile(r"^\s*:\s*")


def _clean_text(text: str) -> str:
    return _LEADING_COLON_RE.sub("", text.strip()).strip()


def _join_body(first_remainder: str, rest_lines: list[str]) -> str:
    first = first_remainder.strip()
    if rest_lines and first:
        return first + "\n" + "\n".join(rest_lines)
    if rest_lines:
        return "\n".join(rest_lines)
    return first


def _build_citation_source(
    opening: Opening, lines: list[str], relative_base: RelativeBase
) -> dict:
    if opening.kind == "relative":
        location = relative_base.citation + render_subdivs(opening.comps or [])
        source_type = relative_base.source_type
    else:
        location = opening.location or ""
        source_type = opening.source_type
    body = _join_body(opening.remainder, lines[1:])
    return {
        "location": location,
        "source_text": _clean_text(body),
        "source_type": source_type,
        "raw": "\n".join(lines).strip(),
    }


def _build_guidance_source(lines: list[str]) -> dict:
    raw = "\n".join(lines).strip()
    first = lines[0].strip()
    location = ""
    body = raw
    if ":" in first:
        candidate = first.split(":", 1)[0].strip()
        # A descriptor, not a whole paragraph.
        if 0 < len(candidate) <= 120:
            location = candidate
            body = raw.split(":", 1)[1]
    return {
        "location": location,
        "source_text": _clean_text(body),
        "source_type": GUIDANCE_SOURCE_TYPE,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_source_field(
    text: str, *, relative_base: RelativeBase = DEFAULT_RELATIVE_BASE
) -> list[dict]:
    """Parse a raw provision string into a list of ``source`` records.

    Each record has keys ``location``, ``source_text``, ``source_type`` and
    ``raw`` (the verbatim segment, kept for auditing/re-parsing). Returns an
    empty list for empty input.
    """
    if not text or not text.strip():
        return []

    segments: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        opening = classify_opening(raw)
        current_opening = current["opening"] if current is not None else None
        if current is None:
            start_new = True
        else:
            start_new = opens_new_provision(opening, current_opening)
        if start_new:
            current = {"opening": opening, "lines": [raw]}
            segments.append(current)
        else:
            current["lines"].append(raw)

    sources: list[dict] = []
    for seg in segments:
        opening = seg["opening"]
        if opening is None:
            sources.append(_build_guidance_source(seg["lines"]))
        else:
            sources.append(_build_citation_source(opening, seg["lines"], relative_base))
    return sources


# ---------------------------------------------------------------------------
# Validation (used by build_benchmark's --check)
# ---------------------------------------------------------------------------

_CANONICAL_LOCATION_RE = re.compile(
    r"^\d+ (U\.S\.C\.|CFR) § \S", re.IGNORECASE
)


def validate_sources(sources: list[dict]) -> list[str]:
    """Return human-readable warnings for low-confidence / incomplete sources."""
    warnings: list[str] = []
    if not sources:
        return ["no source parsed"]
    for i, src in enumerate(sources):
        tag = f"source[{i}]"
        if not src["source_text"]:
            warnings.append(f"{tag}: empty source_text")
        if not src["location"]:
            warnings.append(f"{tag}: empty location")
        if src["source_type"] not in SOURCE_TYPES:
            warnings.append(f"{tag}: unknown source_type {src['source_type']!r}")
        if (
            src["source_type"] != GUIDANCE_SOURCE_TYPE
            and src["location"]
            and not _CANONICAL_LOCATION_RE.match(src["location"])
        ):
            warnings.append(
                f"{tag}: non-canonical location {src['location']!r}"
            )
    return warnings
