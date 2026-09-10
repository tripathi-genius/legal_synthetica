#!/usr/bin/env python3
"""
Evaluate SAVED inference responses against BASE / REFERENCE responses.

This version deliberately does NOT call vLLM and does NOT generate anything.
It compares:
    sft_test_final.jsonl                  = reference/base responses
    inference_*.jsonl                    = saved model responses

The evaluator:
1. Aligns records by id (case_id + task in your files).
2. Aligns assistant turns by their position within the conversation.
3. Computes whole-response semantic cosine similarity.
4. Computes sentence-level precision/recall/F1 using ONE-TO-ONE matching.
5. Computes lexical Jaccard similarity.
6. Computes verdict accuracy/F1 only for verdict_prediction and final_judgment.
7. Produces turn-level, task-level, and overall CSV/JSONL outputs.
8. Performs sanity checks so two different model files cannot silently evaluate
   against the wrong references.

Run with:
    python evaluate_saved_inference.py \
        --reference sft_test_final.jsonl \
        --prediction inference_qwen-3b-finetuned_20260907_193946.jsonl \
        --output_dir evaluation_qwen3b

For Jina:
    --embed_model jinaai/jina-embeddings-v2-base-en

For a smaller/local model:
    --embed_model all-MiniLM-L6-v2
"""

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("saved_inference_evaluator")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    reference_file: str
    prediction_file: str
    output_dir: str = "./evaluation_output"
    embed_model: str = "jinaai/jina-embeddings-v2-base-en"
    similarity_threshold: float = 0.65
    batch_size: int = 32
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Data loading / validation
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


def validate_dataset(records: List[Dict[str, Any]], name: str) -> None:
    if not records:
        raise ValueError(f"{name} is empty.")

    required = {"id", "case_id", "task", "messages"}
    missing = required - set(records[0].keys())
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")

    ids = [r.get("id") for r in records]
    duplicates = [k for k, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(
            f"{name} contains duplicate IDs. First duplicates: {duplicates[:10]}"
        )


def assistant_turns(record: Dict[str, Any]) -> List[str]:
    return [
        str(m.get("content", ""))
        for m in record.get("messages", [])
        if m.get("role") == "assistant"
    ]


def build_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["id"]): r for r in records}


def validate_alignment(
    reference: List[Dict[str, Any]],
    prediction: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:

    ref = build_index(reference)
    pred = build_index(prediction)

    ref_ids = set(ref)
    pred_ids = set(pred)

    missing_predictions = sorted(ref_ids - pred_ids)
    extra_predictions = sorted(pred_ids - ref_ids)

    if missing_predictions:
        raise ValueError(
            f"{len(missing_predictions)} reference records have no prediction. "
            f"Examples: {missing_predictions[:10]}"
        )

    if extra_predictions:
        raise ValueError(
            f"{len(extra_predictions)} prediction records are not in the reference. "
            f"Examples: {extra_predictions[:10]}"
        )

    task_mismatches = []
    case_mismatches = []

    for rid in ref_ids:
        rr = ref[rid]
        pp = pred[rid]

        if rr.get("case_id") != pp.get("case_id"):
            case_mismatches.append(
                (rid, rr.get("case_id"), pp.get("case_id"))
            )

        if rr.get("task") != pp.get("task"):
            task_mismatches.append(
                (rid, rr.get("task"), pp.get("task"))
            )

    if case_mismatches:
        raise ValueError(
            f"case_id mismatch for {len(case_mismatches)} records. "
            f"Examples: {case_mismatches[:5]}"
        )

    if task_mismatches:
        raise ValueError(
            f"task mismatch for {len(task_mismatches)} records. "
            f"Examples: {task_mismatches[:5]}"
        )

    # Turn-count validation is important for your full_chain_multi_turn task.
    turn_mismatches = []
    for rid in sorted(ref_ids):
        nr = len(assistant_turns(ref[rid]))
        npred = len(assistant_turns(pred[rid]))
        if nr != npred:
            turn_mismatches.append((rid, nr, npred))

    if turn_mismatches:
        raise ValueError(
            f"Assistant-turn count mismatch for {len(turn_mismatches)} records. "
            f"Examples: {turn_mismatches[:10]}"
        )

    logger.info("Alignment check passed.")
    logger.info("Reference records: %d", len(reference))
    logger.info("Prediction records: %d", len(prediction))
    logger.info("Common records:    %d", len(ref_ids))

    return ref, pred


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

LEGAL_ABBREVIATIONS = {
    "sec", "art", "ors", "anr", "v", "vs", "no", "nos",
    "hon'ble", "cir", "scc", "scr", "para", "paras", "cl", "cls"
}


def split_sentences(text: str) -> List[str]:
    """
    Lightweight legal-aware sentence splitter.

    It intentionally avoids NLTK resource/version problems and handles the
    common abbreviations in your dataset.
    """
    text = re.sub(r"\r\n?", "\n", str(text)).strip()
    if not text:
        return []

    # Preserve common legal abbreviations temporarily.
    placeholders = {}
    for i, abbr in enumerate(sorted(LEGAL_ABBREVIATIONS, key=len, reverse=True)):
        token = f"__LEGAL_ABBR_{i}__"
        pattern = re.compile(rf"\b{re.escape(abbr)}\.", flags=re.I)
        if pattern.search(text):
            text = pattern.sub(token, text)
            placeholders[token] = abbr + "."

    # Split after sentence punctuation when followed by whitespace.
    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z0-9*#(\[])",
                  "\n", text)

    sentences = []
    for chunk in text.split("\n"):
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if not chunk:
            continue

        # Markdown bullets/headings can contain several logical sentences.
        parts = re.split(r"(?<=[.!?])\s+(?=(?:[-*]|\d+[.)]))", chunk)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    for token, original in placeholders.items():
        sentences = [s.replace(token, original) for s in sentences]

    return sentences


def normalize_for_lexical(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set:
    return set(normalize_for_lexical(text).split())


def jaccard_similarity(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Sentence matching
# ---------------------------------------------------------------------------

def one_to_one_sentence_metrics(
    reference_sentences: List[str],
    prediction_sentences: List[str],
    threshold: float,
    model: SentenceTransformer,
) -> Dict[str, Any]:

    if not reference_sentences and not prediction_sentences:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "matched_pairs": 0,
            "reference_sentences": 0,
            "prediction_sentences": 0,
            "mean_matched_similarity": 1.0,
            "matches": [],
        }

    if not reference_sentences:
        return {
            "tp": 0,
            "fp": len(prediction_sentences),
            "fn": 0,
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "matched_pairs": 0,
            "reference_sentences": 0,
            "prediction_sentences": len(prediction_sentences),
            "mean_matched_similarity": 0.0,
            "matches": [],
        }

    if not prediction_sentences:
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(reference_sentences),
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched_pairs": 0,
            "reference_sentences": len(reference_sentences),
            "prediction_sentences": 0,
            "mean_matched_similarity": 0.0,
            "matches": [],
        }

    ref_emb = model.encode(
        reference_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    pred_emb = model.encode(
        prediction_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sim_matrix = np.matmul(ref_emb, pred_emb.T)

    # Hungarian algorithm gives a TRUE one-to-one alignment.
    rows, cols = linear_sum_assignment(-sim_matrix)

    accepted = []
    rejected = []

    for r, c in zip(rows, cols):
        score = float(sim_matrix[r, c])
        item = {
            "reference_index": int(r),
            "prediction_index": int(c),
            "reference_sentence": reference_sentences[r],
            "prediction_sentence": prediction_sentences[c],
            "similarity": score,
            "accepted": score >= threshold,
        }
        if score >= threshold:
            accepted.append(item)
        else:
            rejected.append(item)

    tp = len(accepted)

    # Any unmatched prediction is FP. A matched-but-below-threshold prediction
    # is also FP because it does not adequately represent its reference.
    fp = (len(prediction_sentences) - len(rows)) + len(rejected)

    # Any unmatched reference is FN. A matched-but-below-threshold reference
    # is also FN.
    fn = (len(reference_sentences) - len(rows)) + len(rejected)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    all_matched_scores = [float(sim_matrix[r, c]) for r, c in zip(rows, cols)]

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_pairs": int(len(rows)),
        "reference_sentences": len(reference_sentences),
        "prediction_sentences": len(prediction_sentences),
        "mean_matched_similarity": (
            float(np.mean(all_matched_scores))
            if all_matched_scores else 0.0
        ),
        "matches": accepted + rejected,
    }


# ---------------------------------------------------------------------------
# Verdict extraction
# ---------------------------------------------------------------------------

def extract_verdict_label(text: str) -> Optional[str]:
    """
    Extract an outcome label when the response actually contains one.

    Priority:
      1. Explicit numeric class: 0/1/2
      2. Explicit outcome phrases
      3. Last clear disposition phrase

    Returns None when the text is too ambiguous.

    IMPORTANT:
    This is not a substitute for a gold structured verdict field. It is only
    used for the two outcome-oriented tasks.
    """
    raw = str(text).strip()
    if not raw:
        return None

    # Your saved verdict_prediction outputs frequently contain just "1.".
    compact = raw.strip().lower()
    numeric = re.fullmatch(r"[\s\-\*]*(?:class\s*)?([012])[\s.)\-:]*", compact)
    if numeric:
        return f"class_{numeric.group(1)}"

    # Explicit labels are preferred.
    explicit_patterns = [
        (r"\b(?:verdict|outcome|result)\s*[:\-]\s*(?:is\s*)?([012])\b",
         lambda m: f"class_{m.group(1)}"),
        (r"\b(?:verdict|outcome|result)\s*[:\-]\s*(allowed|dismissed|partly allowed|partially allowed|acquitted|convicted)\b",
         lambda m: m.group(1).lower()),
    ]

    for pattern, formatter in explicit_patterns:
        m = re.search(pattern, raw, flags=re.I)
        if m:
            return formatter(m)

    # Strong disposition phrases.
    patterns = [
        ("partially_allowed", [
            r"\bpartially\s+(?:allowed|granted)\b",
            r"\bpartly\s+(?:allowed|granted)\b",
        ]),
        ("allowed", [
            r"\bappeal\s+(?:is\s+)?allowed\b",
            r"\bappeal\s+(?:is\s+)?(?:allowed|accepted)\b",
            r"\bpetition\s+(?:is\s+)?allowed\b",
            r"\b(?:order|judgment|conviction)\s+(?:is\s+)?set aside\b",
            r"\bconviction\s+is\s+(?:set aside|quashed)\b",
            r"\bacquitted\b",
        ]),
        ("dismissed", [
            r"\bappeal\s+(?:is\s+)?dismissed\b",
            r"\bpetition\s+(?:is\s+)?dismissed\b",
            r"\bconviction(?:s)?\s+and\s+sentence(?:s)?\s+(?:are\s+)?affirmed\b",
        ]),
        ("convicted", [
            r"\bconvicted\b",
        ]),
    ]

    found = []
    for label, pats in patterns:
        for pat in pats:
            matches = list(re.finditer(pat, raw, flags=re.I))
            for m in matches:
                found.append((m.start(), label))

    if not found:
        return None

    # The disposition near the end of the response is usually the strongest.
    found.sort(key=lambda x: x[0])
    return found[-1][1]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    id: str
    case_id: str
    task: str
    turn_index: int
    reference_text: str
    prediction_text: str
    reference_chars: int
    prediction_chars: int
    semantic_similarity: float
    lexical_jaccard: float
    sentence_precision: float
    sentence_recall: float
    sentence_f1: float
    sentence_tp: int
    sentence_fp: int
    sentence_fn: int
    reference_sentences: int
    prediction_sentences: int
    matched_pairs: int
    mean_matched_similarity: float
    verdict_reference: Optional[str]
    verdict_prediction: Optional[str]
    verdict_correct: Optional[int]


def evaluate(
    reference: List[Dict[str, Any]],
    prediction: List[Dict[str, Any]],
    model: SentenceTransformer,
    threshold: float,
) -> List[TurnResult]:

    ref, pred = validate_alignment(reference, prediction)

    results = []

    for rid in sorted(ref):
        rr = ref[rid]
        pp = pred[rid]

        ref_turns = assistant_turns(rr)
        pred_turns = assistant_turns(pp)

        task = str(rr.get("task", "unknown"))

        for turn_index, (ref_text, pred_text) in enumerate(
            zip(ref_turns, pred_turns)
        ):
            # Whole-response embeddings.
            whole_emb = model.encode(
                [ref_text, pred_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            semantic_similarity = float(np.dot(whole_emb[0], whole_emb[1]))

            ref_sentences = split_sentences(ref_text)
            pred_sentences = split_sentences(pred_text)

            sentence_metrics = one_to_one_sentence_metrics(
                ref_sentences,
                pred_sentences,
                threshold,
                model,
            )

            # Classification is meaningful for outcome-oriented tasks only.
            if task in {"verdict_prediction", "final_judgment"}:
                ref_verdict = extract_verdict_label(ref_text)
                pred_verdict = extract_verdict_label(pred_text)

                if ref_verdict is not None and pred_verdict is not None:
                    verdict_correct = int(ref_verdict == pred_verdict)
                else:
                    verdict_correct = None
            else:
                ref_verdict = None
                pred_verdict = None
                verdict_correct = None

            results.append(
                TurnResult(
                    id=rid,
                    case_id=str(rr.get("case_id", "")),
                    task=task,
                    turn_index=turn_index,
                    reference_text=ref_text,
                    prediction_text=pred_text,
                    reference_chars=len(ref_text),
                    prediction_chars=len(pred_text),
                    semantic_similarity=semantic_similarity,
                    lexical_jaccard=jaccard_similarity(ref_text, pred_text),
                    sentence_precision=sentence_metrics["precision"],
                    sentence_recall=sentence_metrics["recall"],
                    sentence_f1=sentence_metrics["f1"],
                    sentence_tp=sentence_metrics["tp"],
                    sentence_fp=sentence_metrics["fp"],
                    sentence_fn=sentence_metrics["fn"],
                    reference_sentences=sentence_metrics["reference_sentences"],
                    prediction_sentences=sentence_metrics["prediction_sentences"],
                    matched_pairs=sentence_metrics["matched_pairs"],
                    mean_matched_similarity=sentence_metrics["mean_matched_similarity"],
                    verdict_reference=ref_verdict,
                    verdict_prediction=pred_verdict,
                    verdict_correct=verdict_correct,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def aggregate_results(results: List[TurnResult]) -> Tuple[List[Dict], Dict]:
    by_task = defaultdict(list)

    for r in results:
        by_task[r.task].append(r)

    task_rows = []

    for task, rows in sorted(by_task.items()):
        tp = sum(r.sentence_tp for r in rows)
        fp = sum(r.sentence_fp for r in rows)
        fn = sum(r.sentence_fn for r in rows)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        verdict_rows = [
            r for r in rows
            if r.verdict_correct is not None
        ]

        task_rows.append({
            "task": task,
            "records": len({r.id for r in rows}),
            "turns": len(rows),
            "avg_semantic_similarity": safe_mean(
                [r.semantic_similarity for r in rows]
            ),
            "avg_lexical_jaccard": safe_mean(
                [r.lexical_jaccard for r in rows]
            ),
            "sentence_precision_micro": precision,
            "sentence_recall_micro": recall,
            "sentence_f1_micro": f1,
            "sentence_tp": tp,
            "sentence_fp": fp,
            "sentence_fn": fn,
            "verdict_samples": len(verdict_rows),
            "verdict_accuracy": (
                safe_mean([float(r.verdict_correct) for r in verdict_rows])
                if verdict_rows else None
            ),
            "verdict_reference_unparsed": sum(
                r.verdict_reference is None for r in rows
                if r.task in {"verdict_prediction", "final_judgment"}
            ),
            "verdict_prediction_unparsed": sum(
                r.verdict_prediction is None for r in rows
                if r.task in {"verdict_prediction", "final_judgment"}
            ),
        })

    # Overall semantic / lexical metrics are averages over turns.
    all_tp = sum(r.sentence_tp for r in results)
    all_fp = sum(r.sentence_fp for r in results)
    all_fn = sum(r.sentence_fn for r in results)

    overall_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
    overall_recall = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall /
        (overall_precision + overall_recall)
        if overall_precision + overall_recall else 0.0
    )

    verdict_rows = [
        r for r in results
        if r.verdict_correct is not None
    ]

    overall = {
        "records": len({r.id for r in results}),
        "turns": len(results),
        "avg_semantic_similarity": safe_mean(
            [r.semantic_similarity for r in results]
        ),
        "avg_lexical_jaccard": safe_mean(
            [r.lexical_jaccard for r in results]
        ),
        "sentence_precision_micro": overall_precision,
        "sentence_recall_micro": overall_recall,
        "sentence_f1_micro": overall_f1,
        "sentence_tp": all_tp,
        "sentence_fp": all_fp,
        "sentence_fn": all_fn,
        "verdict_samples": len(verdict_rows),
        "verdict_accuracy": (
            safe_mean([float(r.verdict_correct) for r in verdict_rows])
            if verdict_rows else None
        ),
    }

    return task_rows, overall


def compute_verdict_classification_metrics(
    results: List[TurnResult],
) -> List[Dict[str, Any]]:
    """
    Produces explicit classification metrics for outcome tasks.

    Only examples where BOTH gold and prediction can be parsed are used.
    """
    output = []

    for task in ["verdict_prediction", "final_judgment"]:
        rows = [
            r for r in results
            if r.task == task
            and r.verdict_reference is not None
            and r.verdict_prediction is not None
        ]

        if not rows:
            continue

        y_true = [r.verdict_reference for r in rows]
        y_pred = [r.verdict_prediction for r in rows]

        labels = sorted(set(y_true) | set(y_pred))

        output.append({
            "task": task,
            "samples_used": len(rows),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
            "labels": ",".join(labels),
        })

    return output


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    keys = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    output_dir: Path,
    results: List[TurnResult],
    task_rows: List[Dict],
    overall: Dict,
    verdict_rows: List[Dict],
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    turn_path = output_dir / "turn_level_results.csv"
    task_path = output_dir / "task_level_results.csv"
    verdict_path = output_dir / "verdict_classification_results.csv"
    overall_path = output_dir / "overall_summary.json"
    detailed_path = output_dir / "detailed_results.jsonl"

    write_csv(
        turn_path,
        [asdict(r) for r in results],
    )

    write_csv(task_path, task_rows)
    write_csv(verdict_path, verdict_rows)

    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # JSONL keeps the actual reference/prediction pair available for debugging.
    with open(detailed_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    logger.info("Saved:")
    logger.info("  %s", turn_path)
    logger.info("  %s", task_path)
    logger.info("  %s", verdict_path)
    logger.info("  %s", overall_path)
    logger.info("  %s", detailed_path)


def print_summary(
    task_rows: List[Dict[str, Any]],
    overall: Dict[str, Any],
    verdict_rows: List[Dict[str, Any]],
) -> None:

    print("\n" + "=" * 100)
    print("SAVED INFERENCE vs BASE RESPONSE EVALUATION")
    print("=" * 100)

    print(f"Records:                 {overall['records']}")
    print(f"Assistant turns:         {overall['turns']}")
    print(f"Avg semantic similarity: {overall['avg_semantic_similarity']:.4f}")
    print(f"Avg lexical Jaccard:     {overall['avg_lexical_jaccard']:.4f}")
    print(f"Sentence precision:      {overall['sentence_precision_micro']:.4f}")
    print(f"Sentence recall:         {overall['sentence_recall_micro']:.4f}")
    print(f"Sentence F1:             {overall['sentence_f1_micro']:.4f}")

    if overall["verdict_samples"]:
        print(f"Verdict samples:         {overall['verdict_samples']}")
        print(f"Verdict accuracy:        {overall['verdict_accuracy']:.4f}")
    else:
        print("Verdict samples:         0")

    print("\nTASK-LEVEL RESULTS")
    print("-" * 100)

    header = (
        f"{'Task':28} {'N':>5} {'Sim':>8} {'Jaccard':>8} "
        f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'Verdict N':>10}"
    )
    print(header)
    print("-" * 100)

    for row in task_rows:
        verdict_n = row["verdict_samples"]
        print(
            f"{row['task'][:28]:28} "
            f"{row['turns']:5d} "
            f"{row['avg_semantic_similarity']:8.4f} "
            f"{row['avg_lexical_jaccard']:8.4f} "
            f"{row['sentence_precision_micro']:8.4f} "
            f"{row['sentence_recall_micro']:8.4f} "
            f"{row['sentence_f1_micro']:8.4f} "
            f"{verdict_n:10d}"
        )

    if verdict_rows:
        print("\nVERDICT CLASSIFICATION")
        print("-" * 100)
        for row in verdict_rows:
            print(
                f"{row['task']}: "
                f"N={row['samples_used']} "
                f"Accuracy={row['accuracy']:.4f} "
                f"Macro-P={row['precision_macro']:.4f} "
                f"Macro-R={row['recall_macro']:.4f} "
                f"Macro-F1={row['f1_macro']:.4f} "
                f"Labels={row['labels']}"
            )

    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description="Compare saved inference JSONL against base/reference JSONL."
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Base/reference JSONL, e.g. sft_test_final.jsonl",
    )
    parser.add_argument(
        "--prediction",
        required=True,
        help="Saved inference JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation_output",
    )
    parser.add_argument(
        "--embed_model",
        default="jinaai/jina-embeddings-v2-base-en",
        help="SentenceTransformer-compatible embedding model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Sentence semantic-match threshold.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, cpu, or leave unset for SentenceTransformers auto selection.",
    )

    args = parser.parse_args()

    return EvaluationConfig(
        reference_file=args.reference,
        prediction_file=args.prediction,
        output_dir=args.output_dir,
        embed_model=args.embed_model,
        similarity_threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )


def main() -> None:
    config = parse_args()

    reference = load_jsonl(config.reference_file)
    prediction = load_jsonl(config.prediction_file)

    validate_dataset(reference, "Reference dataset")
    validate_dataset(prediction, "Prediction dataset")

    logger.info("Loading embedding model: %s", config.embed_model)

    model_kwargs = {}
    if config.device:
        model_kwargs["device"] = config.device

    model = SentenceTransformer(config.embed_model, **model_kwargs)

    results = evaluate(
        reference=reference,
        prediction=prediction,
        model=model,
        threshold=config.similarity_threshold,
    )

    task_rows, overall = aggregate_results(results)
    verdict_rows = compute_verdict_classification_metrics(results)

    save_outputs(
        output_dir=Path(config.output_dir),
        results=results,
        task_rows=task_rows,
        overall=overall,
        verdict_rows=verdict_rows,
    )

    print_summary(task_rows, overall, verdict_rows)


if __name__ == "__main__":
    main()
