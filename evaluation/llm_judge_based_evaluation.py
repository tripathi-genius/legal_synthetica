#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation for Legal Reasoning Tasks.

Compares:
    --reference  : Base / Gold reference JSONL (e.g. sft_test_final.jsonl)
    --prediction : Model inference JSONL (e.g. inference_*.jsonl)

Tasks Evaluated:
    0. arguments_generation
    1. argument_evaluation
    2. final_judgment
    3. full_chain_single_turn
    4. full_chain_multi_turn (evaluated turn-by-turn based on stage)
    5. verdict_prediction

Outputs:
    - turn_level_results.csv
    - task_level_results.csv
    - detailed_results.jsonl
    - overall_summary.json
"""

import argparse
import csv
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("llm_judge_evaluator")

# ---------------------------------------------------------------------------
# Task-Specific LLM Judge Prompts & Rubrics (Scale 1-10)
# ---------------------------------------------------------------------------

PROMPTS = {
    "arguments_generation": """You are an expert appellate court judge evaluating an AI assistant trained to generate legal arguments for Indian appellate cases.
Compare the PREDICTED ARGUMENTS against the GOLD REFERENCE ARGUMENTS using the provided CASE FACTS.

[CASE FACTS & STATUTORY CONTEXT]
{context}

[GOLD REFERENCE ARGUMENTS]
{reference}

[PREDICTED ARGUMENTS TO EVALUATE]
{prediction}

[EVALUATION RUBRIC: 1 to 10]
Evaluate across these dimensions:
1. Factual & Statutory Fidelity: Are arguments strictly grounded in the given record without hallucinating facts or authorities?
2. Bilateral Comprehensiveness: Are strong, viable points formulated for BOTH sides (Prosecution/State and Defense/Appellant)?
3. Alignment with Gold Standard: Does the prediction capture the primary legal levers present in the reference?

Scoring Guide:
- 1-2: Fabricated facts/statutes; missing one entire side.
- 3-4: Plausible prose but ignores essential statutory provisions or key facts.
- 5-6: Moderately grounded; covers basic arguments but misses nuanced legal points.
- 7-8: Strong, legally sound arguments for both sides closely aligning with the reference.
- 9-10: Exemplary appellate-grade legal arguments, fully grounded and complete.

Respond ONLY with a JSON object in this exact format:
{{
  "score": <integer from 1 to 10>,
  "sub_scores": {{
    "factual_fidelity": <1-10>,
    "bilateral_balance": <1-10>,
    "legal_acuity": <1-10>
  }},
  "reasoning": "<Concise 2-3 sentence legal justification for the score>"
}}
""",

    "argument_evaluation": """You are an expert appellate court judge evaluating an AI assistant's judicial evaluation of legal arguments.
The model must evaluate which arguments to ACCEPT and which to REJECT, explaining the judicial rationale behind each determination.

[CASE FACTS & ARGUMENTS]
{context}

[GOLD REFERENCE EVALUATION]
{reference}

[PREDICTED EVALUATION TO EVALUATE]
{prediction}

[EVALUATION RUBRIC: 1 to 10]
Evaluate across these dimensions:
1. Soundness of Acceptance/Rejection: Does the judge correctly identify which arguments withstand scrutiny versus which collapse under law?
2. Doctrinal Consistency: Does the reasoning apply the correct statutory tests, burdens of proof, and binding precedents?
3. Alignment with Gold Standard: Does the analytical conclusion agree with the reference judicial analysis?

Scoring Guide:
- 1-2: Completely irrational or contradictory determinations; misstates basic law.
- 3-4: Superficial evaluation; accepts flawed arguments or rejects valid statutory claims without legal basis.
- 5-6: Acceptable judicial logic, but misses core statutory/precedential tests found in the reference.
- 7-8: Rigorous judicial reasoning; accept/reject determinations match reference with strong justification.
- 9-10: Masterful judicial evaluation exhibiting deep legal scrutiny and exact alignment.

Respond ONLY with a JSON object in this exact format:
{{
  "score": <integer from 1 to 10>,
  "sub_scores": {{
    "judicial_logic": <1-10>,
    "doctrinal_accuracy": <1-10>,
    "determination_alignment": <1-10>
  }},
  "reasoning": "<Concise 2-3 sentence legal justification for the score>"
}}
""",

    "final_judgment": """You are an expert appellate court judge evaluating the final holding and disposition of an Indian appellate case.

[CASE FACTS & EVALUATION]
{context}

[GOLD REFERENCE HOLDING & RESULT]
{reference}

[PREDICTED HOLDING & RESULT TO EVALUATE]
{prediction}

[EVALUATION RUBRIC: 1 to 10]
Evaluate across these dimensions:
1. Disposition & Outcome Correctness: Is the bottom-line order correct (e.g., appeal allowed, dismissed, conviction set aside, remanded, compensation/decree adjusted)?
2. Decisive Ratio Decidendi: Is the core legal reason driving the holding correct and grounded in the material?
3. Relief & Direction Accuracy: Are the specific directions (costs, sentences, bail cancellation, interest, restitution) properly stated?

Scoring Guide:
- 1-2: Opposite disposition (e.g., affirms conviction when it was quashed); flawed premise.
- 3-4: Correct disposition keyword but fundamentally mistaken rationale or inconsistent relief.
- 5-6: Correct bottom-line outcome, but key remedial details or decisive ratios differ from the reference.
- 7-8: Accurate disposition and strong legal ratio matching the reference with minor phrasing variance.
- 9-10: Perfect legal disposition, exact decisive rationale, and accurate directions.

Respond ONLY with a JSON object in this exact format:
{{
  "score": <integer from 1 to 10>,
  "verdict_match": <true or false>,
  "sub_scores": {{
    "disposition_accuracy": <1-10>,
    "ratio_correctness": <1-10>,
    "relief_directions": <1-10>
  }},
  "reasoning": "<Concise 2-3 sentence legal justification for the score>"
}}
""",

    "verdict_prediction": """You are an expert appellate court judge evaluating a short 1-3 sentence verdict and core reason prediction.

[CASE MATERIAL & SUBMISSIONS]
{context}

[GOLD REFERENCE VERDICT]
{reference}

[PREDICTED VERDICT TO EVALUATE]
{prediction}

[EVALUATION RUBRIC: 1 to 10]
1. Outcome Correctness: Did the model accurately predict the final appellate disposition?
2. Decisive Ground: Did the model pinpoint the single decisive ground that dictated the result?

Scoring Guide:
- 1-2: Wrong bottom-line verdict (e.g., allowed vs dismissed).
- 3-4: Ambiguous outcome; unclear whether the appeal succeeds or fails.
- 5-6: Correct outcome, but the stated reason is legally erroneous or missing.
- 7-8: Correct outcome with a sound, legally relevant decisive reason closely tracking the reference.
- 9-10: Flawless prediction of the bottom line and exact core legal ratio.

Respond ONLY with a JSON object in this exact format:
{{
  "score": <integer from 1 to 10>,
  "verdict_match": <true or false>,
  "sub_scores": {{
    "outcome_match": <1-10>,
    "core_reason_accuracy": <1-10>
  }},
  "reasoning": "<Concise 2-3 sentence legal justification for the score>"
}}
""",

    "full_chain_single_turn": """You are an expert appellate judge evaluating a complete 4-stage legal reasoning trace:
1. Arguments FOR (Prosecution/State)
2. Arguments AGAINST (Defense/Appellant)
3. Evaluation (Reasoning to Accept & Reject)
4. Final Judgment (Holding & Disposition)

[CASE FACTS & PROVISIONS]
{context}

[GOLD REFERENCE TRACE]
{reference}

[PREDICTED FULL CHAIN TO EVALUATE]
{prediction}

[EVALUATION RUBRIC: 1 to 10]
1. Structural Completeness: Are all 4 stages clearly articulated and developed?
2. Internal Coherence: Does stage 3 follow from stages 1-2, and does stage 4 follow strictly from stage 3?
3. Substantive & Dispositional Accuracy: How faithfully does the entire trace align with the reference analysis and result?

Scoring Guide:
- 1-2: Missing major stages; hallucinated record; incorrect outcome.
- 3-4: Disjointed chain; stages contradict one another; outcome unsupported.
- 5-6: Complete 4 stages, but reasoning is superficial or outcome partially deviates.
- 7-8: Highly coherent, comprehensive trace with sound evaluations and correct disposition.
- 9-10: Exemplary judicial opinion drafting matching gold standard depth and accuracy.

Respond ONLY with a JSON object in this exact format:
{{
  "score": <integer from 1 to 10>,
  "verdict_match": <true or false>,
  "sub_scores": {{
    "structural_completeness": <1-10>,
    "internal_coherence": <1-10>,
    "disposition_accuracy": <1-10>
  }},
  "reasoning": "<Concise 2-3 sentence legal justification for the score>"
}}
"""
}

# ---------------------------------------------------------------------------
# Data Loading & Verification
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def validate_alignment(
    reference: List[Dict[str, Any]],
    prediction: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    ref = {str(r["id"]): r for r in reference}
    pred = {str(r["id"]): r for r in prediction}

    ref_ids = set(ref)
    pred_ids = set(pred)

    if ref_ids != pred_ids:
        missing = sorted(ref_ids - pred_ids)
        extra = sorted(pred_ids - ref_ids)
        raise ValueError(
            f"Dataset alignment error: {len(missing)} missing, {len(extra)} extra."
        )

    for rid in ref_ids:
        if ref[rid].get("task") != pred[rid].get("task"):
            raise ValueError(f"Task mismatch for ID {rid}: {ref[rid].get('task')} vs {pred[rid].get('task')}")

    logger.info("Alignment validated: %d shared records.", len(ref_ids))
    return ref, pred


# ---------------------------------------------------------------------------
# Turn Pair Extractor
# ---------------------------------------------------------------------------

@dataclass
class TurnEvalItem:
    id: str
    case_id: str
    task: str
    turn_index: int
    prompt_context: str
    reference_text: str
    prediction_text: str
    judge_prompt_type: str


def prepare_eval_items(
    ref_map: Dict[str, Dict[str, Any]],
    pred_map: Dict[str, Dict[str, Any]],
) -> List[TurnEvalItem]:
    items = []

    for rid, r_rec in sorted(ref_map.items()):
        p_rec = pred_map[rid]
        task = r_rec.get("task", "unknown")
        case_id = r_rec.get("case_id", "")

        r_msgs = r_rec.get("messages", [])
        p_msgs = p_rec.get("messages", [])

        # Extract assistant responses and preceding user context
        r_assistant = [m["content"] for m in r_msgs if m.get("role") == "assistant"]
        p_assistant = [m["content"] for m in p_msgs if m.get("role") == "assistant"]

        # Collect user prompts
        user_prompts = [m["content"] for m in r_msgs if m.get("role") == "user"]
        base_context = user_prompts[0] if user_prompts else ""

        for turn_idx, (r_text, p_text) in enumerate(zip(r_assistant, p_assistant)):
            # Determine appropriate prompt rubric
            if task == "full_chain_multi_turn":
                # Turn 0: Arguments, Turn 1: Evaluation, Turn 2: Final Judgment
                if turn_idx == 0:
                    judge_type = "arguments_generation"
                elif turn_idx == 1:
                    judge_type = "argument_evaluation"
                else:
                    judge_type = "final_judgment"
                
                # For context, provide previous turns if available
                turn_context = user_prompts[turn_idx] if turn_idx < len(user_prompts) else base_context
            else:
                judge_type = task if task in PROMPTS else "full_chain_single_turn"
                turn_context = base_context

            items.append(
                TurnEvalItem(
                    id=rid,
                    case_id=case_id,
                    task=task,
                    turn_index=turn_idx,
                    prompt_context=turn_context,
                    reference_text=r_text,
                    prediction_text=p_text,
                    judge_prompt_type=judge_type,
                )
            )

    return items


# ---------------------------------------------------------------------------
# LLM Judge Worker
# ---------------------------------------------------------------------------

def parse_judge_json(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    else:
        bracket_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if bracket_match:
            cleaned = bracket_match.group(1)

    try:
        data = json.loads(cleaned)
        score = float(data.get("score", 0))
        # Clamp score between 1 and 10
        score = max(1.0, min(10.0, score))
        data["score"] = score
        return data
    except Exception:
        # Fallback regex extraction if JSON is malformed
        score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
        score = float(score_match.group(1)) if score_match else 5.0
        return {
            "score": max(1.0, min(10.0, score)),
            "reasoning": cleaned[:300],
            "sub_scores": {},
            "verdict_match": None,
        }


def call_judge_api(client: OpenAI, model: str, prompt: str, max_retries: int = 4) -> str:
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return res.choices[0].message.content or ""
        except Exception as err:
            logger.warning("Judge call attempt %d failed: %s", attempt + 1, err)
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def evaluate_item(
    item: TurnEvalItem,
    client: OpenAI,
    model: str,
) -> Dict[str, Any]:
    template = PROMPTS.get(item.judge_prompt_type, PROMPTS["full_chain_single_turn"])
    judge_prompt = template.format(
        context=item.prompt_context,
        reference=item.reference_text,
        prediction=item.prediction_text,
    )

    try:
        raw_judgment = call_judge_api(client, model, judge_prompt)
        parsed = parse_judge_json(raw_judgment)
    except Exception as exc:
        logger.error("Failed to judge ID %s turn %d: %s", item.id, item.turn_index, exc)
        parsed = {
            "score": 1.0,
            "reasoning": f"Evaluation call failed: {exc}",
            "sub_scores": {},
            "verdict_match": None,
        }
        raw_judgment = str(exc)

    return {
        "id": item.id,
        "case_id": item.case_id,
        "task": item.task,
        "turn_index": item.turn_index,
        "judge_prompt_type": item.judge_prompt_type,
        "score": parsed.get("score", 1.0),
        "verdict_match": parsed.get("verdict_match"),
        "sub_scores": json.dumps(parsed.get("sub_scores", {})),
        "reasoning": parsed.get("reasoning", ""),
        "reference_chars": len(item.reference_text),
        "prediction_chars": len(item.prediction_text),
        "raw_judgment": raw_judgment,
    }


# ---------------------------------------------------------------------------
# Aggregation & Output
# ---------------------------------------------------------------------------

def aggregate_scores(results: List[Dict[str, Any]]) -> Tuple[List[Dict], Dict]:
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)

    task_rows = []
    for task, rows in sorted(by_task.items()):
        scores = [r["score"] for r in rows]
        verdict_rows = [r["verdict_match"] for r in rows if r["verdict_match"] is not None]

        task_rows.append({
            "task": task,
            "records": len({r["id"] for r in rows}),
            "turns_evaluated": len(rows),
            "mean_score_1_to_10": round(float(np.mean(scores)), 3),
            "std_score": round(float(np.std(scores)), 3),
            "min_score": round(float(np.min(scores)), 3),
            "max_score": round(float(np.max(scores)), 3),
            "verdict_accuracy": (
                round(float(np.mean(verdict_rows)), 3) if verdict_rows else None
            ),
        })

    all_scores = [r["score"] for r in results]
    all_verdicts = [r["verdict_match"] for r in results if r["verdict_match"] is not None]

    overall = {
        "total_records": len({r["id"] for r in results}),
        "total_turns": len(results),
        "overall_mean_score_1_to_10": round(float(np.mean(all_scores)), 3),
        "overall_std_score": round(float(np.std(all_scores)), 3),
        "overall_verdict_accuracy": (
            round(float(np.mean(all_verdicts)), 3) if all_verdicts else None
        ),
    }

    return task_rows, overall


def save_reports(
    output_dir: Path,
    turn_results: List[Dict[str, Any]],
    task_rows: List[Dict[str, Any]],
    overall: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    turn_path = output_dir / "turn_level_results.csv"
    task_path = output_dir / "task_level_results.csv"
    overall_path = output_dir / "overall_summary.json"
    detailed_path = output_dir / "detailed_results.jsonl"

    # 1. Turn level CSV
    fieldnames = [
        "id", "case_id", "task", "turn_index", "judge_prompt_type",
        "score", "verdict_match", "sub_scores", "reasoning",
        "reference_chars", "prediction_chars"
    ]
    with open(turn_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(turn_results)

    # 2. Task level CSV
    if task_rows:
        with open(task_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
            writer.writeheader()
            writer.writerows(task_rows)

    # 3. Overall JSON
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # 4. Detailed JSONL
    with open(detailed_path, "w", encoding="utf-8") as f:
        for r in turn_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Saved all evaluation files into %s", output_dir)


def print_summary(task_rows: List[Dict[str, Any]], overall: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("                 LLM-AS-A-JUDGE EVALUATION RESULTS (Scale 1-10)                 ")
    print("=" * 90)
    print(f"Total Unique Cases  : {overall['total_records']}")
    print(f"Total Turns Judged  : {overall['total_turns']}")
    print(f"Overall Average Score: {overall['overall_mean_score_1_to_10']:.2f} / 10.0")
    if overall["overall_verdict_accuracy"] is not None:
        print(f"Verdict Match Rate   : {overall['overall_verdict_accuracy'] * 100:.1f}%")
    print("-" * 90)
    header = f"{'Task':28} {'Turns':>6} {'Mean Score':>12} {'Std':>8} {'Min':>6} {'Max':>6} {'Verdict Acc':>13}"
    print(header)
    print("-" * 90)
    for r in task_rows:
        vacc = f"{r['verdict_accuracy']*100:.1f}%" if r['verdict_accuracy'] is not None else "N/A"
        print(
            f"{r['task'][:28]:28} "
            f"{r['turns_evaluated']:6d} "
            f"{r['mean_score_1_to_10']:12.2f} "
            f"{r['std_score']:8.2f} "
            f"{r['min_score']:6.1f} "
            f"{r['max_score']:6.1f} "
            f"{vacc:>13}"
        )
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Main Execution CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Judge Evaluator for Indian Appellate Legal Reasoning"
    )
    parser.add_argument("--reference", required=True, help="Path to gold reference JSONL")
    parser.add_argument("--prediction", required=True, help="Path to predicted inference JSONL")
    parser.add_argument("--output_dir", default="./llm_judge_output", help="Directory for output reports")
    parser.add_argument("--model", default="gpt-oss-120b", help="LLM judge model name")
    parser.add_argument("--base_url", default="http://api.llm.apps.os.dcs.gla.ac.uk/v1", help="API Base URL")
    parser.add_argument("--concurrency", type=int, default=20, help="Parallel execution workers")

    args = parser.parse_args()

    api_key = os.environ.get("IDA_LLM_API_KEY", "EMPTY")
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    ref_data = load_jsonl(args.reference)
    pred_data = load_jsonl(args.prediction)

    ref_map, pred_map = validate_alignment(ref_data, pred_data)
    eval_items = prepare_eval_items(ref_map, pred_map)

    logger.info("Starting LLM evaluation of %d items across %d threads...", len(eval_items), args.concurrency)

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_item = {
            executor.submit(evaluate_item, item, client, args.model): item
            for item in eval_items
        }

        completed = 0
        for future in as_completed(future_to_item):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 10 == 0 or completed == len(eval_items):
                logger.info("Progress: %d / %d items evaluated", completed, len(eval_items))

    # Sort results by ID and turn index
    results.sort(key=lambda x: (x["id"], x["turn_index"]))

    task_rows, overall = aggregate_scores(results)
    save_reports(Path(args.output_dir), results, task_rows, overall)
    print_summary(task_rows, overall)


if __name__ == "__main__":
    main()
