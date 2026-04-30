import csv
from collections import defaultdict
from pathlib import Path

csv_path = Path(__file__).parent / "results.csv"

questions = ["q1_recognizes_followup", "q2_correct_yes_no", "q3_includes_context"]

counts = defaultdict(lambda: {"yes": 0, "total": 0})

with open(csv_path) as f:
    for row in csv.DictReader(f):
        for q in questions:
            key = (row["model"], q)
            counts[key]["total"] += 1
            if row[q] == "YES":
                counts[key]["yes"] += 1

models = sorted({k[0] for k in counts})

for model in models:
    print(f"\n{model}")
    for q in questions:
        key = (model, q)
        yes = counts[key]["yes"]
        total = counts[key]["total"]
        pct = 100 * yes / total if total else 0
        print(f"  {q}: {yes}/{total} ({pct:.0f}%)")
