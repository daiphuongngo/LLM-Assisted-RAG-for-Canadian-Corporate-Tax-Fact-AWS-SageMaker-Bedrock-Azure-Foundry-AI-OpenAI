from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    label = data.get("summary", {}).get("label", path.stem)
    rows = {r["id"]: r for r in data.get("rows", [])}
    return label, data.get("summary", {}), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--output-dir", default="benchmark_outputs")
    ap.add_argument("--label", default="comparison")
    args = ap.parse_args()

    loaded = [load(Path(p)) for p in args.runs]
    all_ids = sorted(set().union(*(set(rows.keys()) for _, _, rows in loaded)))
    comparison = {"summaries": {label: summary for label, summary, _ in loaded}, "questions": []}
    for qid in all_ids:
        row = {"id": qid, "runs": {}}
        for label, _, rows in loaded:
            r = rows.get(qid, {})
            row["runs"][label] = {
                "answer": r.get("answer"),
                "page_hit": r.get("page_hit"),
                "contains_expected": r.get("contains_expected"),
                "value_hit": r.get("value_hit"),
                "total_seconds": r.get("total_seconds"),
            }
        comparison["questions"].append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{args.label}.json"
    tp = out_dir / f"{args.label}.txt"
    jp.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    with tp.open("w", encoding="utf-8") as f:
        f.write("SUMMARIES\n")
        f.write(json.dumps(comparison["summaries"], indent=2, ensure_ascii=False))
        f.write("\n\nPER QUESTION\n")
        for q in comparison["questions"]:
            f.write("=" * 100 + "\n")
            f.write(f"{q['id']}\n")
            for label, r in q["runs"].items():
                f.write(f"[{label}] page_hit={r['page_hit']} contains={r['contains_expected']} value_hit={r['value_hit']} seconds={r['total_seconds']}\n")
                f.write(f"Answer: {r['answer']}\n")
    print(f"Saved: {jp}")
    print(f"Saved: {tp}")


if __name__ == "__main__":
    main()
