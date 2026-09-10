import os

# ============================================================
# MEMORY / TOKENIZER ENVIRONMENT
# Must be set before importing torch
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from liger_kernel.transformers import apply_liger_kernel_to_qwen2


# ============================================================
# CONFIG
# ============================================================

model_id = "Qwen/Qwen2.5-3B-Instruct"

data_file = "sft_train_synthetic_micro_case.jsonl"

output_dir = "./qwen_legal_finetune"

MAXIMUM_MODEL_LENGTH = 7_100


# ============================================================
# LOAD DATASET
# ============================================================

dataset = load_dataset(
    "json",
    data_files=data_file,
    split="train",
)

print(
    f"Loaded dataset. Total cases: {len(dataset):,}"
)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
)

# Qwen uses EOS as PAD when a separate PAD token
# is not defined.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# TOKENIZATION
# ============================================================

def preprocess_function(example):

    messages = example["messages"]

    # --------------------------------------------------------
    # Full conversation
    # --------------------------------------------------------

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # --------------------------------------------------------
    # Tokenize complete conversation
    # --------------------------------------------------------

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=MAXIMUM_MODEL_LENGTH,
    )

    input_ids = encoded["input_ids"]

    attention_mask = encoded["attention_mask"]

    # --------------------------------------------------------
    # Default: ignore every token for loss.
    #
    # We will enable loss ONLY for assistant messages.
    # --------------------------------------------------------

    labels = [-100] * len(input_ids)

    # --------------------------------------------------------
    # Identify assistant portions
    # --------------------------------------------------------

    conversation = []

    for message in messages:

        if message["role"] == "assistant":

            # Conversation before this assistant response
            before_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            before_tokens = tokenizer(
                before_text,
                add_special_tokens=False,
                truncation=True,
                max_length=MAXIMUM_MODEL_LENGTH,
                padding=False,
            )["input_ids"]

            start_position = min(
                len(before_tokens),
                len(input_ids),
            )

            # Conversation including this assistant response
            conversation_with_answer = (
                conversation + [message]
            )

            answer_text = tokenizer.apply_chat_template(
                conversation_with_answer,
                tokenize=False,
                add_generation_prompt=False,
            )

            answer_tokens = tokenizer(
                answer_text,
                add_special_tokens=False,
                truncation=True,
                max_length=MAXIMUM_MODEL_LENGTH,
                padding=False,
            )["input_ids"]

            end_position = min(
                len(answer_tokens),
                len(input_ids),
            )

            # ------------------------------------------------
            # Only assistant tokens contribute to loss.
            # ------------------------------------------------

            for i in range(
                start_position,
                end_position,
            ):
                labels[i] = input_ids[i]

        conversation.append(message)

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,

        # Metadata
        "id": example["id"],
        "case_id": example["case_id"],
        "task": example["task"],
    }


# ============================================================
# TOKENIZE DATASET
# ============================================================

tokenized_dataset = dataset.map(
    preprocess_function,

    batched=False,

    remove_columns=[
        "messages",
    ],

    num_proc=16,

    desc="Tokenizing dataset",
)

print(
    f"Dataset successfully tokenized. "
    f"Total cases: {len(tokenized_dataset):,}"
)


# ============================================================
# REMOVE EXAMPLES WITHOUT ASSISTANT TOKENS
# ============================================================

tokenized_dataset = tokenized_dataset.filter(
    lambda x: (
        len(x["input_ids"]) > 1
        and any(
            label != -100
            for label in x["labels"]
        )
    ),
    num_proc=16,
)

print(
    f"Cases after filtering: "
    f"{len(tokenized_dataset):,}"
)


# ============================================================
# SEQUENCE LENGTH STATISTICS
# ============================================================

lengths = [
    len(x["input_ids"])
    for x in tokenized_dataset
]

print("=" * 60)
print("SEQUENCE LENGTH STATISTICS")
print("=" * 60)

print(
    f"Mean: {np.mean(lengths):,.0f}"
)

print(
    f"P50 : {np.percentile(lengths, 50):,.0f}"
)

print(
    f"P90 : {np.percentile(lengths, 90):,.0f}"
)

print(
    f"P95 : {np.percentile(lengths, 95):,.0f}"
)

print(
    f"P99 : {np.percentile(lengths, 99):,.0f}"
)

print(
    f"Max : {np.max(lengths):,}"
)

print("=" * 60)


# ============================================================
# APPLY LIGER KERNEL
# ============================================================

apply_liger_kernel_to_qwen2(
    rope=True,
    rms_norm=True,
    swiglu=True,
    fused_linear_cross_entropy=True,
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
    return_tensors="pt",
)


# ============================================================
# MODEL LOADING
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    model_id,

    # A6000 supports BF16.
    torch_dtype=torch.bfloat16,

    # Memory-efficient attention for 7.1k context.
    attn_implementation="flash_attention_2",

    # Required for gradient checkpointing.
    use_cache=False,
)


# ============================================================
# LoRA
# ============================================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(
    model,
    peft_config,
)

model.print_trainable_parameters()


# ============================================================
# GRADIENT CHECKPOINTING
# ============================================================

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    }
)

model.enable_input_require_grads()


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=output_dir,

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    per_device_train_batch_size=14,

    gradient_accumulation_steps=2,

    # With 3 GPUs:
    #
    # 1 × 8 × 3 = 24
    #
    # effective global batch size.
    #

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_train_epochs=3,

    learning_rate=2e-4,

    weight_decay=0.01,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    max_grad_norm=1.0,

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    bf16=True,

    tf32=True,

    # --------------------------------------------------------
    # Gradient checkpointing
    # --------------------------------------------------------

    gradient_checkpointing=True,

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optim="adamw_torch_fused",

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=10,

    logging_first_step=True,

    report_to="none",

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=2,

    save_safetensors=True,

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader_num_workers=4,

    dataloader_pin_memory=True,

    dataloader_persistent_workers=True,

    dataloader_prefetch_factor=2,

    # --------------------------------------------------------
    # Group similar sequence lengths together.
    #
    # This is particularly useful for 7,100-token sequences.
    # --------------------------------------------------------

    group_by_length=True,

    # --------------------------------------------------------
    # DDP
    # --------------------------------------------------------

    ddp_find_unused_parameters=False,

    # --------------------------------------------------------
    # Dataset columns
    # --------------------------------------------------------

    remove_unused_columns=True,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="no",

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=42,

    data_seed=42,

    # --------------------------------------------------------
    # Hub
    # --------------------------------------------------------

    push_to_hub=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator,
)


# ============================================================
# TRAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STARTING QWEN2.5-3B LEGAL SFT")
    print("=" * 70)

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    effective_batch_size = (
        1
        * 8
        * world_size
    )

    print(
        f"World size: {world_size}"
    )

    print(
        f"Effective global batch size: "
        f"{effective_batch_size}"
    )

    print(
        f"Maximum sequence length: "
        f"{MAXIMUM_MODEL_LENGTH}"
    )

    print(
        "Precision: BF16"
    )

    print(
        "Attention: Flash Attention 2"
    )

    print(
        "Gradient checkpointing: ON"
    )

    print(
        "LoRA: r=16, alpha=32"
    )

    print(
        "Loss: assistant-only"
    )

    print("=" * 70)

    trainer.train()

    # ========================================================
    # SAVE FINAL LORA ADAPTER
    # ========================================================

    final_output_dir = os.path.join(
        output_dir,
        "final_adapter",
    )

    trainer.save_model(
        final_output_dir
    )

    tokenizer.save_pretrained(
        final_output_dir
    )

    print("=" * 70)
    print("TRAINING COMPLETE")
    print(
        f"Adapter saved to: {final_output_dir}"
    )
    print("=" * 70)