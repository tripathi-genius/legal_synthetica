# An Introduction to Fine-Tuning

Fine-tuning is the process of taking a large pre-trained language model (like Llama 3 or Qwen 2.5) and adapting it to perform specific tasks or behave in a certain way by training it on a smaller, specialized dataset. Instead of teaching the model how to understand language from scratch—which takes massive computing power and time—fine-tuning builds upon the model's existing knowledge.

### The Approach: Supervised Fine-Tuning (SFT) with LoRA

The scripts provided (`train_llama_8b.py`, `train_llama3b.py`, and `train_qwen.py`) use a specific, highly efficient method for fine-tuning:

1.  **Supervised Fine-Tuning (SFT):** The model is trained on examples of exact inputs and desired outputs (in this case, legal conversation examples). The loss function is specifically calculated only on the *assistant's* responses, teaching the model how to reply to user prompts.
2.  **Low-Rank Adaptation (LoRA):** Instead of updating all billions of parameters in the model (Full Fine-Tuning), LoRA freezes the original model weights and injects small, trainable "adapter" modules. This drastically reduces the memory required to train the model, making it possible to fine-tune large models on standard consumer GPUs.
3.  **Liger Kernel & Flash Attention:** These are advanced optimization techniques used in the scripts to further reduce memory consumption and speed up the processing of very long text sequences (up to 7,100 tokens).

---

## Required Libraries

To run the fine-tuning scripts successfully, you need a specific software environment. Here is the list of required Python libraries and what they do in the context of your training pipeline:

*   **`torch`**: The core deep learning framework (PyTorch) used to handle the neural network calculations and GPU memory management.
*   **`numpy`**: Used for fast mathematical operations and calculating sequence length statistics during the tokenization phase.
*   **`datasets`**: A Hugging Face library used to efficiently load, process, and map the `sft_train_synthetic_micro_case.jsonl` data file.
*   **`transformers`**: The main Hugging Face library used to load the pre-trained models, the tokenizers, and the `Trainer` class which orchestrates the training loop.
*   **`peft`**: Parameter-Efficient Fine-Tuning. This library provides the `LoraConfig` and handles injecting the LoRA adapters into the base models.
*   **`liger-kernel`**: A specialized library that applies efficient kernels to Hugging Face models (like Llama and Qwen), replacing standard operations with highly optimized ones to save memory and increase throughput.
*   **`flash-attn`**: FlashAttention-2 is a highly optimized attention algorithm that drastically speeds up training and reduces the memory footprint, especially for long context windows.
*   **`wandb`**: Weights & Biases. Required by `train_llama3b.py` to log training metrics, track runs, and monitor the fine-tuning progress in the cloud.

### Installation Command

You can install all of these dependencies via pip:

```bash
pip install torch numpy datasets transformers peft liger-kernel wandb flash-attn
```