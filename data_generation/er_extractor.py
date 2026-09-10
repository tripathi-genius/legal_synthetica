import os
import re
import json
import pickle
import random
import requests
from openai import OpenAI
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. ENDPOINTS & MODEL POOLS
# ==========================================
LOCAL_ENDPOINTS = [
    "http://0.0.0.0:8000/generate",
    # "http://0.0.0.0:8001/generate"
]

remote_client = OpenAI(
    base_url="http://api.llm.apps.os.dcs.gla.ac.uk/v1", 
    api_key="ida_IH9MGTzw96irxPvltv6neNlVt5rksafDAsNJmTur"
)

REMOTE_EXTRACTION_MODELS = [
    "gpt-oss-20b",
    "mistral-small-3.2-24b-2506",
    "qwen-3-32b",
    "glm-4.7-flash"
]

# ==========================================
# 2. SYSTEM PROMPTS
# ==========================================
mapper_system_prompt = """<Objective>
You are an expert Legal Data Extraction AI preparing structured logic maps for a Logic Tracing Engine. 
Your task is to analyze the provided legal document chunk, extract key entities, and rigorously map the logical relationships between them.
</Objective>

<Definitions>
1. Entities:
   - Fact: Objective events, timelines, or data points.
   - Claim: Subjective assertions, arguments, or allegations.
   - Precedent: Prior case law, judicial rulings, or binding decisions cited.
   - Act: Statutes, legislation, constitutional articles, or formal policies.
2. Relationships:
   - Types: "proves", "contradicts", "triggers", "relies_upon", "interprets".
</Definitions>

<Extraction Rules>
1. ID Generation: Entity IDs MUST be strictly unique, lowercase, alphanumeric, and use underscores (e.g., `fact_1`). 
2. Relationship Integrity: Every `source_id` and `target_id` MUST exactly match an `id` defined in the entities array.
3. Strict JSON: Output ONLY valid JSON enclosed in a ```json ``` markdown code block. No conversational filler.
</Extraction Rules>

<Output Schema>
{
  "entities": [
    {
      "id": "unique_string_id",
      "type": "Fact | Claim | Precedent | Act",
      "description": "Concise, 1-2 sentence description",
      "attribution": "Entity author/speaker"
    }
  ],
  "relationships": [
    {
      "source_id": "id of the source entity",
      "target_id": "id of the target entity",
      "relationship_type": "proves | contradicts | triggers | relies_upon | interprets",
      "rationale_snippet": "Verbatim short quote proving this connection",
      "status": "Contested | Accepted by Court | Implicit Assumption | Explicit Mandate"
    }
  ]
}
</Output Schema>"""

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def extract_json(response_text):
    if not response_text:
        raise ValueError("Empty response from model")
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response_text)
    fence = re.search(r"```json\s*([\s\S]*?)\s*```", cleaned)
    if fence: return json.loads(fence.group(1))
    brace = re.search(r"\{[\s\S]*\}", cleaned)
    if brace: return json.loads(brace.group(0))
    raise ValueError("No JSON object found in model response")

def perform_extraction(chunk_text):
    extract_prompt = f"{mapper_system_prompt}\n\n---\n\nExtract relationships for this text:\n\n{chunk_text}"
    targets = LOCAL_ENDPOINTS + REMOTE_EXTRACTION_MODELS
    random.shuffle(targets) 
    
    last_err = None
    for target in targets:
        try:
            if target.startswith("http"):
                payload = {"prompt": extract_prompt, "temperature": 0.1, "stream": False}
                res = requests.post(target, json=payload, timeout=300)
                res.raise_for_status()
                raw_json_str = res.json().get("text", "")
            else:
                res = remote_client.chat.completions.create(
                    model=target,
                    messages=[{"role": "user", "content": extract_prompt}],
                    temperature=0.1
                )
                raw_json_str = res.choices[0].message.content

            parsed_json = extract_json(raw_json_str)
            if isinstance(parsed_json, dict):
                parsed_json["_meta"] = {"extractor_used": target, "status": "unverified"}
            return parsed_json
        except Exception as e:
            last_err = e
            continue 
    raise RuntimeError(f"All extraction targets failed. Last error: {last_err}")

def process_single_chunk(item):
    purpose = item.get("taxonomy", {}).get("section_purpose", "").lower()
    chunk_text = item.get("text", "")
    item["chunk_text_source"] = chunk_text
    
    if any(k in purpose for k in ["fact", "argument", "precedent", "analysis", "ruling", "citations"]):
        try:
            item["logic_map"] = perform_extraction(chunk_text)
        except Exception as e:
            item["logic_map"] = {"error": str(e), "entities": [], "relationships": []}
    else:
        item["logic_map"] = {"entities": [], "relationships": []}
    return item

# ==========================================
# 4. EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    INPUT_FILE = "task_class_saved.pickle"
    OUTPUT_FILE = "extracted_chunks_raw.json"
    print(f"Loading data from {INPUT_FILE}...")
    
    with open(INPUT_FILE, "rb") as f:
        judgments_data = pickle.load(f)
    print("Successfully loaded as Pickle.")
    
    MAX_WORKERS = 200
    print(f"\n--- Running Distributed ER Extraction on {len(judgments_data)} chunks ---")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_chunk, item): item for item in judgments_data}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Chunks"):
            try:
                future.result() 
            except Exception as e:
                print(f"Thread error: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(judgments_data, f, indent=4)
    print(f"\n[SAVED] Raw extraction maps -> '{OUTPUT_FILE}'")
