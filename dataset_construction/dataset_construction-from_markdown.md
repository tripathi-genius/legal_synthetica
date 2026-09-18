# Legal Case SFT Dataset Builder

## What This Program Does (Function)

This tool is designed to process a large text corpus of legal cases (specifically formatted Indian criminal appeals) and convert it into a Supervised Fine-Tuning (SFT) dataset. This dataset is formatted as conversational turns (System, User, Assistant) to train Large Language Models (LLMs) on complex judicial reasoning.

The program consists of three main components:
1. **`parsing.py`**: Scans the raw Markdown corpus, identifies individual cases using headers, cleans up the text (handling unicode and formatting), and extracts specific structural sections (Facts, Law, Arguments, Logic, Conclusion).
2. **`tasks.py`**: Maps the parsed case sections into six distinct machine learning training tasks (e.g., Argument Generation, Verdict Prediction, Full Chain Reasoning). It formats these tasks into standard message arrays.
3. **`build_dataset.py`**: The main execution script. It orchestrates the parsing and task generation, randomly splits the cases into training and validation sets, and writes the final outputs.

## How to Run It

### Prerequisites
- **Python 3.7+** is required.
- The script relies entirely on standard Python libraries (`re`, `json`, `argparse`, `pathlib`, `dataclasses`, `unicodedata`), so no external dependencies via `pip` are necessary.

### Basic Command

To run the program, open your terminal and execute `build_dataset.py` using Python. You must provide two required arguments: the path to your input Markdown file and the folder where you want to save the results.

```bash
python build_dataset.py --input path/to/your/corpus.md --output-dir ./dataset_output
```

### Advanced Options

You can customize the dataset generation using several optional flags:

- `--val-ratio`: The fraction of cases to hold out for the validation set (default is `0.02` or 2%).
- `--seed`: Set a random seed for reproducible train/validation splitting (default is `42`).
- `--tasks`: Specify exactly which tasks to generate if you don't want all six. Options include `arguments_generation`, `argument_evaluation`, `final_judgment`, `full_chain_single_turn`, `full_chain_multi_turn`, and `verdict_prediction`.
- `--preview-n`: The number of cases to render into a human-readable preview file (default is `3`).

**Example with custom options:**
```bash
python build_dataset.py \
    --input corpus.md \
    --output-dir ./my_model_data \
    --val-ratio 0.10 \
    --seed 123 \
    --tasks arguments_generation verdict_prediction
```

## Expected Output

After running successfully, the program will generate the following files in your specified `--output-dir`:

1. `sft_train.jsonl`: The main training dataset in JSON Lines format.
2. `sft_val.jsonl`: The validation dataset used for evaluating the model.
3. `parse_report.json`: A detailed summary of the parsing process, including counts of complete/incomplete cases and the distribution of tasks.
4. `sample_preview.md`: A human-readable markdown file showing a few generated examples so you can verify the output quality before training.