# Legal Reasoning Models Evaluation

This repository contains two primary scripts to evaluate the performance of Large Language Models (LLMs) on Indian appellate legal reasoning tasks. The evaluation is split into two methodologies: an LLM-as-a-Judge approach and a Semantic Similarity approach.

---

## 1. LLM-as-a-Judge Evaluation

**File:** `llm_judge_based_evaluation.py`

### Brief Explanation
This script uses a strong LLM to act as an expert appellate court judge, evaluating the quality of model-generated predictions against gold-standard references. It assesses tasks like argument generation, argument evaluation, final judgment, and full-chain legal reasoning. 

It constructs task-specific prompts and asks the evaluator LLM to score the prediction on a scale of 1-10 based on factual fidelity, doctrinal consistency, and alignment with the gold reference. It outputs detailed reasoning and sub-scores for each turn.

### Prerequisites
*   Python 3.x
*   `openai` and `numpy` libraries (`pip install openai numpy`)
*   An API endpoint running your evaluator model (e.g., via vLLM).
*   Environment variable `IDA_LLM_API_KEY` set (if authentication is required by your endpoint).

### How to Run
Use the following commands to evaluate the different pre-trained and fine-tuned models. The script requires the reference JSONL and the prediction JSONL.

**Evaluate Llama 8B Base:**
```bash
python llm_judge_based_evaluation.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_llama-8b-base_20260907_175712.jsonl --output_dir llama8b_base_eval 
```

**Evaluate Qwen 3B Base:**
```bash
python llm_judge_based_evaluation.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_qwen-3b-base_20260907_165829.jsonl --output_dir qwen3b_base_eval 
```

**Evaluate Llama 3B Fine-tuned:**
```bash
python llm_judge_based_evaluation.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_llama-3b-finetuned_20260907_192207.jsonl --output_dir llama3b_finetune_eval 
```

**Evaluate Llama 8B Fine-tuned:**
```bash
python llm_judge_based_evaluation.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_llama-8b-finetuned_20260907_202421.jsonl --output_dir llama8b_finetuned_eval 
```

**Evaluate Qwen 3B Fine-tuned:**
```bash
python llm_judge_based_evaluation.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_qwen-3b-finetuned_20260907_193946.jsonl --output_dir qwen3b_finetuned_eval 
```

---

## 2. Semantic Evaluation

**File:** `save_semantic_eval.py`

### Brief Explanation
Unlike the LLM Judge, this script performs a deterministic, reference-based evaluation using embedding models to measure semantic similarity. It does *not* call an LLM API to generate text. 

Instead, it:
1. Calculates whole-response cosine similarity between the reference and prediction.
2. Uses the Hungarian algorithm (linear sum assignment) to match individual sentences and compute Precision, Recall, and F1 scores.
3. Calculates lexical Jaccard similarity.
4. Extracts outcome labels for verdict prediction and final judgment tasks to measure raw classification accuracy.

### Prerequisites
*   Python 3.x
*   `sentence-transformers`, `scipy`, `scikit-learn`, `numpy` libraries (`pip install sentence-transformers scipy scikit-learn numpy`)

### How to Run
You can run this script using the default embedding model (`jinaai/jina-embeddings-v2-base-en`) or specify a smaller local model.

**Basic Run (Default Settings):**
```bash
python save_semantic_eval.py \
    --reference test_micro_cases/sft_test_final.jsonl \
    --prediction inference_qwen-3b-finetuned_20260907_193946.jsonl \
    --output_dir semantic_evaluation_qwen3b
```

**Run with a smaller, faster embedding model:**
```bash
python save_semantic_eval.py \
    --reference test_micro_cases/sft_test_final.jsonl \
    --prediction inference_llama-8b-finetuned_20260907_202421.jsonl \
    --output_dir semantic_evaluation_llama8b \
    --embed_model all-MiniLM-L6-v2
```

---

## Outputs

Both scripts will generate a suite of output files in the specified `--output_dir`:
*   `turn_level_results.csv`: Fine-grained metrics for every individual turn/message evaluated.
*   `task_level_results.csv`: Aggregated metrics grouped by task type (e.g., arguments_generation, verdict_prediction).
*   `overall_summary.json`: High-level summary of the entire evaluation run.
*   `detailed_results.jsonl`: Comprehensive record of the evaluation including raw outputs, matched text, and sub-scores.
*   `verdict_classification_results.csv`: (Semantic Eval only) Specific macro/micro classification metrics for verdict-oriented tasks.