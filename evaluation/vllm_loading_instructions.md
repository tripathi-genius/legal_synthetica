# Serving Models with vLLM (Base vs. LoRA)

vLLM is a highly optimized library for LLM inference and serving. This guide covers how to load your pre-trained base models and your fine-tuned LoRA models using both the vLLM Command Line Interface (for serving as an API) and the Python API (for offline inference).

## 1. Loading Base Models (Without LoRA)

When serving a standard base model, you just need to point vLLM to the model repository or local directory.

### Option A: Using the vLLM API Server (CLI)
This launches an OpenAI-compatible API server.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/base_model \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```
*(Example: Replace `/path/to/base_model` with `meta-llama/Meta-Llama-3-8B` or your local Qwen/Llama path).*

### Option B: Offline Inference (Python Script)
Use this if you are running inference in a batch script without needing a persistent API.

```python
from vllm import LLM, SamplingParams

# Load the base model
llm = LLM(model="/path/to/base_model")

# Generate text
prompts = ["Can you explain the verdict in the recent case?"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

---

## 2. Loading Fine-Tuned Models (With LoRA)

vLLM supports dynamically loading LoRA adapters on top of base models without needing to permanently merge the weights beforehand.

### Option A: Using the vLLM API Server (CLI)
To use LoRA, you must explicitly enable it with the `--enable-lora` flag and specify your adapters using `--lora-modules`.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/base_model \
    --enable-lora \
    --lora-modules legal_lora=/path/to/lora_adapter \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```
* **`--model`**: The path to the foundational model (e.g., Llama-8b-base).
* **`--lora-modules`**: Format is `name=path`. In requests to this server, you would pass `"model": "legal_lora"` in your API payload to route traffic through the adapter.

### Option B: Offline Inference (Python Script)
In Python, you enable LoRA during model initialization and attach the specific LoRA request to your generation call.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 1. Initialize the LLM with LoRA enabled
llm = LLM(
    model="/path/to/base_model", 
    enable_lora=True, 
    max_lora_rank=64  # Adjust to match your LoRA's rank (r value)
)

prompts = ["Can you explain the verdict in the recent case?"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

# 2. Pass the LoRA request during generation
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest(
        lora_name="legal_finetune",
        lora_int_id=1,
        lora_local_path="/path/to/lora_adapter"
    )
)

for output in outputs:
    print(output.outputs[0].text)
```

### Important Notes on vLLM LoRA:
* **Max LoRA Rank**: If you get errors related to LoRA rank sizes in Python, ensure you set `max_lora_rank` in the `LLM()` constructor to match or exceed the `r` value used during your PEFT/LoRA training (e.g., 16, 32, 64).
* **Multiple LoRAs**: You can load multiple different LoRA adapters simultaneously on the same base model by passing multiple comma-separated pairs to `--lora-modules` or using different `lora_int_id` numbers in Python.