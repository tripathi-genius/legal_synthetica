#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

from parsing import load_and_parse_all
from tasks import build_examples_for_case, TASK_BUILDERS


ALL_TASKS = list(TASK_BUILDERS.keys())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to the raw markdown corpus")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs into")
    p.add_argument("--val-ratio", type=float, default=0.02,
                    help="Fraction of CASES (not rows) held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS,
                    help="Which task types to generate (default: all six)")
    p.add_argument("--preview-n", type=int, default=3,
                    help="How many example cases to render into sample_preview.md")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Reading {in_path} ...")
    full_text = in_path.read_text(encoding="utf-8")

    print("[2/5] Splitting and parsing cases ...")
    parsed_cases, parse_stats = load_and_parse_all(full_text)
    print(f"      found {parse_stats['total_found']} case(s); "
          f"{parse_stats['complete']} complete, {parse_stats['incomplete']} incomplete")

    print(f"[3/5] Building SFT examples for tasks: {', '.join(args.tasks)} ...")
    all_examples = []
    per_task_counts = {t: 0 for t in args.tasks}
    per_case_counts = {}
    for pc in parsed_cases:
        exs = build_examples_for_case(pc, args.tasks)
        per_case_counts[pc.case_id] = len(exs)
        for ex in exs:
            per_task_counts[ex["task"]] += 1
        all_examples.extend(exs)
    print(f"      generated {len(all_examples)} total SFT rows "
          f"from {len(parsed_cases)} cases")

    print(f"[4/5] Splitting train/val by case_id (val_ratio={args.val_ratio}) ...")
    case_ids = sorted({pc.case_id for pc in parsed_cases})
    rng = random.Random(args.seed)
    rng.shuffle(case_ids)
    if not case_ids or args.val_ratio <= 0:
        n_val_cases = 0
    else:
        n_val_cases = max(1, int(len(case_ids) * args.val_ratio))
        n_val_cases = min(n_val_cases, len(case_ids) - 1)  # always keep >=1 case for train
    val_case_ids = set(case_ids[:n_val_cases])
    train_examples = [e for e in all_examples if e["case_id"] not in val_case_ids]
    val_examples = [e for e in all_examples if e["case_id"] in val_case_ids]
    rng.shuffle(train_examples)
    rng.shuffle(val_examples)

    print("[5/5] Writing outputs ...")
    train_path = out_dir / "sft_train.jsonl"
    val_path = out_dir / "sft_val.jsonl"
    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    report = {
        "input_file": str(in_path),
        "cases_found": parse_stats["total_found"],
        "cases_complete": parse_stats["complete"],
        "cases_incomplete": parse_stats["incomplete"],
        "missing_section_counts": parse_stats["missing_section_counts"],
        "incomplete_case_ids_sample": parse_stats["incomplete_case_ids"][:25],
        "tasks_generated": args.tasks,
        "examples_per_task": per_task_counts,
        "total_examples": len(all_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "val_case_count": len(val_case_ids),
        "train_case_count": len(case_ids) - len(val_case_ids),
        "seed": args.seed,
    }
    report_path = out_dir / "parse_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    preview_path = out_dir / "sample_preview.md"
    write_preview(preview_path, parsed_cases[: args.preview_n], all_examples)

    print()
    print("Done.")
    print(f"  train : {train_path}  ({len(train_examples)} rows)")
    print(f"  val   : {val_path}  ({len(val_examples)} rows)")
    print(f"  report: {report_path}")
    print(f"  preview: {preview_path}")
    print()
    print("Examples per task:")
    for t, c in per_task_counts.items():
        print(f"    {t:26s} {c}")
    if parse_stats["incomplete"]:
        print()
        print(f"WARNING: {parse_stats['incomplete']} case(s) were missing at least one "
              f"section and were partially skipped -- see parse_report.json "
              f"('missing_section_counts', 'incomplete_case_ids_sample').")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview(path: Path, cases, all_examples) -> None:
    lines = ["# Sample SFT examples (human-readable preview)\n"]
    for pc in cases:
        case_examples = [e for e in all_examples if e["case_id"] == pc.case_id]
        lines.append(f"\n---\n\n## Case `{pc.case_id}`\n")
        for ex in case_examples:
            lines.append(f"\n### Task: `{ex['task']}`  (`{len(ex['messages'])}` messages)\n")
            for m in ex["messages"]:
                role = m["role"]
                content = m["content"]
                shown = content if len(content) < 1200 else content[:1200] + " …[truncated for preview]"
                lines.append(f"\n**{role}:**\n\n{shown}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
