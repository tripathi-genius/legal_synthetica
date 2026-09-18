# Legal Synthetica: Synthetic Data Generation for Hierarchical Legal Reasoning 

This repository is designed for **synthetic data generation for hierarchical reasoning**. It provides a complete, end-to-end framework for processing complex legal documents, creating structured synthetic datasets, fine-tuning Large Language Models (LLMs), and rigorously evaluating their analytical capabilities.

Below is a simple guide to understanding the workflow of this repository and the corresponding documentation files you can reference for each step.

## 1. Data Processing and Synthetic Generation
The foundation of this project is the data pipeline, which is thoroughly documented in `pipeline_documentation.md`. This pipeline breaks down raw legal judgments to build both training and testing datasets. It operates in three distinct stages:
* **Chunking & Taxonomy:** Splits judgment texts into smaller chunks and extracts key topics and taxonomies.
* **Entity-Relationship Extraction:** Maps logical connections between facts, claims, precedents, and acts.
* **Micro Case Generation:** Builds cohesive synthetic narratives and deep reasoning frameworks to be used for model training.

## 2. Model Fine-Tuning
Once the synthetic data is generated, it is used to train language models. You can refer to `fine_tuning_overview.md` for a complete breakdown of this process.
* The repository uses Supervised Fine-Tuning (SFT) combined with Low-Rank Adaptation (LoRA) to efficiently adapt large models (like Llama 3 and Qwen 2.5) on standard consumer GPUs.
* Advanced techniques like Flash Attention and Liger Kernel are implemented to manage memory when processing long sequences of legal text.

## 3. Serving Models
After fine-tuning, models can be served for inference following the instructions in `vllm_loading_instructions.md`.
* The guide covers how to load standard base models or dynamically inject your new LoRA adapters using the highly optimized vLLM library.
* Inference can be run either through an OpenAI-compatible API server or offline via a Python script.

## 4. Evaluation and Visualization
To measure how well the models perform hierarchical reasoning on legal tasks, refer to `evaluation_instructions.md`. The repository evaluates performance using two primary methods:
* **LLM-as-a-Judge:** Uses a strong evaluator LLM to score model predictions based on factual fidelity, doctrinal consistency, and alignment with references.
* **Semantic Evaluation:** Uses embedding models to measure deterministic semantic similarities, linear sum assignments, and raw classification accuracy without generating new text.

To visualize these evaluation metrics, you can use the scripts outlined in `plot_generation_overview.md`, which generate graphs like Jaccard similarity plots and LLM judge score charts without altering the underlying data.

## 5. Quick Start: Downloading Pre-Trained Models
If you prefer to skip the data generation and fine-tuning phases entirely, you can download the final, fine-tuned models directly. Instructions and secure links for the fine-tuned Llama and Qwen models are provided in `model_download_instructions.md`
