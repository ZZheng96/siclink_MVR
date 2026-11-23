"""Collect experiment JSON files into a compact task-level OCF leaderboard."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def candidates(result):
    evaluation = result.get("evaluation", {})
    rows = []
    if "ocf_mean" in evaluation:
        rows.append(("independent", evaluation))
    for variant in ("direct", "retrieval"):
        if variant in evaluation:
            rows.append((variant, evaluation[variant]))
    assignment = result.get("evaluation_assignment")
    if isinstance(assignment, dict) and "ocf_mean" in assignment:
        rows.append(("assignment", assignment))
    return rows


def main():
    rows = []
    for path in sorted((ROOT / "test").glob("*/results_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("experiment") == "00_sanity":
            continue
        task = result.get("config", {}).get("task")
        if task is None and str(result.get("experiment", "")).startswith(("01", "05", "07")):
            task = "task1"
        if task not in ("task1", "task2", "task3"):
            continue
        for variant, metrics in candidates(result):
            rows.append({
                "task": task,
                "experiment": result.get("experiment"),
                "name": result.get("name", path.stem),
                "variant": variant,
                "ocf_mean": metrics["ocf_mean"],
                "top1_accuracy": metrics.get("top1_accuracy"),
                "n": metrics.get("n"),
                "source": str(path.relative_to(ROOT)),
            })
    best = {}
    for task in ("task1", "task2", "task3"):
        task_rows = [row for row in rows if row["task"] == task]
        if task_rows:
            best[task] = max(task_rows, key=lambda row: row["ocf_mean"])
    composite = sum(row["ocf_mean"] for row in best.values()) / len(best) if best else None
    output = {
        "metric": "official OCF on fixed local validation subsets; not a Kaggle score",
        "leaderboard_rank_1_snapshot": 54.01486,
        "comparable_to_hidden_leaderboard": False,
        "best_by_task": best,
        "equal_task_composite": composite,
        "gap_to_rank_1_snapshot": None if composite is None else 54.01486 - composite,
        "candidates": sorted(rows, key=lambda row: (row["task"], -row["ocf_mean"], row["name"])),
    }
    path = ROOT / "test/results_summary.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best_by_task": best, "equal_task_composite": composite, "gap": output["gap_to_rank_1_snapshot"]}, indent=2))


if __name__ == "__main__":
    main()
