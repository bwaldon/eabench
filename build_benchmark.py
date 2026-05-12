"""Convert per-item plain-text benchmark items into a single benchmark.jsonl.

Reads items/<n>/full_item.txt, detects section headers (which vary across the
50 items), and emits one JSON record per item with normalized field names.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ITEMS_DIR = ROOT / "items"
OUT_FILE = ROOT / "benchmark.jsonl"

# Ordered list of (header_regex, canonical_field). Order matters: more
# specific / longer patterns must come before shorter prefixes that would
# otherwise match them (e.g., "query continuation" before "query").
HEADER_TO_FIELD: list[tuple[str, str]] = [
    (r"relevant\s+statutory\s+provisions?", "statutory_provision"),
    (r"statutory\s+provisions?", "statutory_provision"),
    (r"relevant\s+provisions?(?:\s*\(emphasis\s+added\))?", "statutory_provision"),
    (r"legislative\s+provisions?", "statutory_provision"),
    (r"relevant", "statutory_provision"),
    (r"expected\s+follow-?up\s+question", "expected_followup"),
    (r"query\s+continuation(?:\s*\([^)]*\))?", "query_continuation"),
    (r"follow\s*up\s+query", "query_continuation"),
    (r"user\s+query", "user_query"),
    (r"query", "user_query"),
    (r"expected\s+final\s+answer", "expected_final_answer"),
    (r"expected\s+answer", "expected_final_answer"),
    (r"expected\s+response(?:\s*\([^)]*\))?", "expected_final_answer"),
    (r"final\s+output", "expected_final_answer"),
]

FIELD_ORDER = [
    "user_query",
    "statutory_provision",
    "expected_followup",
    "query_continuation",
    "expected_final_answer",
]


def match_header(line: str) -> tuple[str | None, str]:
    """Return (canonical_field, remainder_on_same_line) if line is a section
    header, otherwise (None, "").
    """
    for pattern, field in HEADER_TO_FIELD:
        m = re.fullmatch(
            r"\s*" + pattern + r"\s*:?\s*(.*?)\s*", line, re.IGNORECASE
        )
        if m:
            return field, m.group(1)
    return None, ""


def parse_full_item(text: str) -> dict[str, str]:
    """Split text into sections keyed by canonical field name.

    Content that appears before the first recognized header is assigned to
    `statutory_provision` (some items lead with an unlabeled provision).
    """
    sections: dict[str, list[str]] = {}
    current: str | None = "statutory_provision"
    buf: list[str] = []

    def flush() -> None:
        if current is None:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        sections.setdefault(current, []).append(content)

    for raw in text.splitlines():
        field, rest = match_header(raw)
        if field is not None:
            flush()
            current = field
            buf = [rest] if rest else []
        else:
            buf.append(raw)
    flush()

    return {k: "\n\n".join(v) for k, v in sections.items()}


def build_record(item_dir: Path) -> dict:
    full_item_path = item_dir / "full_item.txt"
    user_query_path = item_dir / "user_query.txt"

    full_text = full_item_path.read_text(encoding="utf-8")
    parsed = parse_full_item(full_text)

    user_query = parsed.get("user_query", "").strip()
    if (not user_query) and user_query_path.exists():
        user_query = user_query_path.read_text(encoding="utf-8").strip()

    record = {
        "task_id": item_dir.name,
        "user_query": user_query,
        "statutory_provision": parsed.get("statutory_provision", "").strip(),
        "expected_followup": parsed.get("expected_followup", "").strip(),
        "query_continuation": parsed.get("query_continuation", "").strip(),
        "expected_final_answer": parsed.get("expected_final_answer", "").strip(),
        "gold_exchange": full_text.strip(),
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_FILE,
        help=f"Output JSONL path (default: {OUT_FILE})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report items with missing fields and exit non-zero if any.",
    )
    args = parser.parse_args()

    item_dirs = sorted(
        (d for d in ITEMS_DIR.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    records = [build_record(d) for d in item_dirs]

    missing: list[tuple[str, list[str]]] = []
    for rec in records:
        gaps = [f for f in FIELD_ORDER if not rec.get(f)]
        if gaps:
            missing.append((rec["task_id"], gaps))

    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {args.out}")
    if missing:
        print(f"\nItems with missing fields ({len(missing)}):", file=sys.stderr)
        for task_id, gaps in missing:
            print(f"  {task_id}: missing {gaps}", file=sys.stderr)
        if args.check:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
