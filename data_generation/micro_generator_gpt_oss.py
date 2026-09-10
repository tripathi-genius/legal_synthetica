import os
import json
import time
import random
from collections import defaultdict, deque
from tqdm.auto import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

random.seed(42)
DATA_FILE = 'extracted_er_with_chunks.json'
TARGET_CASE_COUNT = 2500
OUTPUT_HF_DATASET = 'micro_cases_hf_stf_format.jsonl'
OUTPUT_HF_DATASET_JSON = 'sft_micro_cases_2.json'
OUTPUT_MD_FILE = 'micro_cases_and_args.md'
LOG_FILE = 'logs_micro.json'

# API Setup
BASE_URL = "http://api.llm.apps.os.dcs.gla.ac.uk/v1"
if 'IDA_LLM_API_KEY' not in os.environ:
    raise ValueError("Please set the IDA_LLM_API_KEY environment variable.")

client = OpenAI(base_url=BASE_URL, api_key=os.environ['IDA_LLM_API_KEY'])

# Multi-Model Configuration
MODEL_MICRO_CASE = "gpt-oss-20b"
MODEL_ARGUMENTS = "gpt-oss-120b"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
CONCURRENT_REQUESTS = 10  

# Token & Context Limits
CHARS_PER_TOKEN = 3.5
MAX_MICRO_CASE_CONTEXT = 115_000  # Updated from 160_000
MAX_ARGUMENTS_CONTEXT = 64_000    # Updated from 32_000

# Global State
api_log_queue = deque(maxlen=10)
log_lock = threading.Lock()
main_lock = threading.Lock()


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def estimate_tokens(text):
    """Estimates the number of tokens in a string based on character count."""
    return int(len(str(text)) / CHARS_PER_TOKEN)


def log_api_interaction(prompt_messages, response_text, model_used, error=None):
    """Updates the log file with the latest request/response safely across threads."""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_used,
        "prompt": json.dumps(prompt_messages, indent=2),
        "response": response_text,
        "error": error
    }
    
    with log_lock:
        api_log_queue.append(log_entry)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(list(api_log_queue), f, indent=4, ensure_ascii=False)


def call_model(messages, model_name, max_tokens=4000, temperature=0.3):
    """Calls the external API using the specific model requested."""
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=300 
            )
            
            output = response.choices[0].message.content.strip()
            
            if not output:
                raise ValueError(f"{model_name} returned an empty string.")
                
            log_api_interaction(messages, output, model_name)
            return output
            
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                
    log_api_interaction(messages, None, model_name, error=last_error)
    raise RuntimeError(f"{model_name} call failed after {MAX_RETRIES} tries: {last_error}")


def group_and_reconstruct_cases(file_path):
    """
    Reads the extracted JSON, groups by case_id, and returns a sorted list 
    of individual chunks with their specific texts and entities.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    cases = defaultdict(list)
    for record in records:
        case_id = record.get('id')
        if case_id:
            cases[case_id].append(record)
            
    processed_cases = {}
    for case_id, chunks in cases.items():
        chunks.sort(key=lambda x: x.get('chunk_id', 0))
        
        formatted_chunks = []
        for chunk in chunks:
            text = chunk.get('text', '')
            logic = chunk.get('logic_map', {})
            entities = logic.get('entities', []) if logic else []
            
            formatted_chunks.append({
                "chunk_id": chunk.get('chunk_id', 0),
                "text": text,
                "entities": entities
            })
            
        processed_cases[case_id] = formatted_chunks
        
    return processed_cases


# ==========================================
# PROMPT BUILDERS
# ==========================================

def get_fact_extraction_messages(chunk_text, entities, current_summary=None):
    """Builds a rolling history to extract a single, unified fact pattern across chunks."""
    system_prompt = (
        "You are an expert legal analyst. Your task is to extract objective facts from the provided text to build a 'Micro Case' narrative.\n"
        "Do not include legal arguments or conclusions—only the core actors, dispute, and timeline."
    )
    
    if current_summary:
        system_prompt += "\n\nYou are continuing an ongoing summary. Seamlessly integrate the NEW facts into the EXISTING narrative to output one cohesive text."
        
    messages = [{"role": "system", "content": system_prompt}]
    
    if current_summary:
        messages.append({
            "role": "user", 
            "content": f"EXISTING NARRATIVE:\n{current_summary}\n\nNEW TEXT TO INTEGRATE:\nEntities: {json.dumps(entities)}\nText: {chunk_text}\n\nUpdate and output the complete, unified narrative:"
        })
    else:
        messages.append({
            "role": "user", 
            "content": f"Entities: {json.dumps(entities)}\n\nText: {chunk_text}\n\nExtract and output the initial facts:"
        })
        
    return messages


def get_deep_reasoning_messages(final_micro_case):
    """Prompts the secondary model to generate the structured reasoning framework."""
    system_prompt = (
        "You are a master appellate judge and legal scholar.\n"
        "Based on the factual 'Micro Case' provided, generate a comprehensive legal reasoning framework.\n\n"
        "You MUST structure your output exactly with these headers in Markdown:\n"
        "### 1. Relevant Sections of Law\n"
        "### 2. Arguments FOR (Prosecution/Plaintiff)\n"
        "### 3. Arguments AGAINST (Defense/Defendant)\n"
        "### 4. Judicial Acceptance Logic (Why to accept arguments)\n"
        "### 5. Judicial Rejection Logic (Why to dismiss arguments, using high-level bench logic)\n"
        "### 6. Final Conclusion / Judgment\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Micro Case Facts:\n{final_micro_case}\n\nGenerate the legal reasoning framework:"}
    ]


# ==========================================
# THREAD WORKER
# ==========================================

def process_single_case(case_id, original_chunks):
    """Processes a list of pre-existing logical chunks, maintaining conversational state."""
    try:
        case_sft_examples = []
        current_summary = None
        
        # Output token buffers
        MODEL_MICRO_CASE_MAX_OUT = 4000
        MODEL_ARGUMENTS_MAX_OUT = 15000
        
        # Phase 1: Iterative Fact Building
        for i, chunk in enumerate(original_chunks):
            fact_messages = get_fact_extraction_messages(chunk["text"], chunk.get("entities", []), current_summary)
            
            # Context Limit Check for Micro Case Model
            prompt_tokens = estimate_tokens(json.dumps(fact_messages))
            if prompt_tokens + MODEL_MICRO_CASE_MAX_OUT > MAX_MICRO_CASE_CONTEXT:
                return False, case_id, f"Micro Case model context exceeded on chunk {i} ({prompt_tokens} tokens)", None, None
            
            fact_output = call_model(fact_messages, model_name=MODEL_MICRO_CASE, max_tokens=MODEL_MICRO_CASE_MAX_OUT, temperature=0.3)
            
            if not fact_output:
                return False, case_id, f"Micro Case model failed on chunk {i}", None, None
                
            # Update the running narrative for the next chunk
            current_summary = fact_output
            
            # Save the fact-extraction step for SFT
            case_sft_examples.append({
                "case_id": case_id,
                "step": f"fact_extraction_chunk_{i}",
                "messages": fact_messages + [{"role": "assistant", "content": fact_output}]
            })
            
        # Phase 2: Deep Reasoning Generation
        reasoning_messages = get_deep_reasoning_messages(current_summary)
        
        # Context Limit Check for Arguments Model
        reasoning_prompt_tokens = estimate_tokens(json.dumps(reasoning_messages))
        if reasoning_prompt_tokens + MODEL_ARGUMENTS_MAX_OUT > MAX_ARGUMENTS_CONTEXT:
            return False, case_id, f"Arguments model context exceeded ({reasoning_prompt_tokens} tokens prompt + {MODEL_ARGUMENTS_MAX_OUT} buffer > {MAX_ARGUMENTS_CONTEXT} max)", None, None

        reasoning_output = call_model(reasoning_messages, model_name=MODEL_ARGUMENTS, max_tokens=MODEL_ARGUMENTS_MAX_OUT, temperature=0.3)
        
        if not reasoning_output:
            return False, case_id, "Arguments model failed to generate reasoning", None, None
            
        # Save the final reasoning step for SFT
        case_sft_examples.append({
            "case_id": case_id,
            "step": "deep_reasoning_generation",
            "messages": reasoning_messages + [{"role": "assistant", "content": reasoning_output}]
        })
        
        # Format the combined layout for Markdown documentation
        formatted_md = f"## Micro Case Facts\n{current_summary}\n\n{reasoning_output}"
        
        return True, case_id, None, case_sft_examples, formatted_md

    except Exception as e:
        return False, case_id, str(e), None, None


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Reading and mapping original case chunks...")
    all_cases = group_and_reconstruct_cases(DATA_FILE)
    
    unique_case_ids = list(all_cases.keys())
    print(f"Total unique cases found: {len(unique_case_ids)}")
    random.shuffle(unique_case_ids)
    
    target_successes = min(TARGET_CASE_COUNT, len(unique_case_ids))
    print(f"Attempting to generate {target_successes} valid reasoning cases concurrently...")

    hf_training_data = []
    successful_markdowns = []
    failed_cases = []
    
    success_count = 0
    executor = ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS)
    
    futures = {
        executor.submit(process_single_case, cid, all_cases[cid]): cid 
        for cid in unique_case_ids
    }

    try:
        with tqdm(total=target_successes, desc="Successful Cases") as pbar:
            for future in as_completed(futures):
                with main_lock:
                    if success_count >= target_successes:
                        break

                try:
                    success, case_id, error_msg, sft_data, md_text = future.result()
                except Exception as e:
                    success = False
                    case_id = futures[future]
                    error_msg = str(e)

                if success:
                    with main_lock:
                        if success_count < target_successes:
                            hf_training_data.extend(sft_data)
                            successful_markdowns.append((case_id, md_text))
                            success_count += 1
                            pbar.update(1)
                            
                            if success_count >= target_successes:
                                break
                else:
                    with main_lock:
                        failed_cases.append({"case_id": case_id, "error": error_msg})

    finally:
        print("\nTarget reached or loop finished. Terminating remaining workers...")
        executor.shutdown(wait=False, cancel_futures=True)

    print(f"\nSaving {len(hf_training_data)} HF training examples...")
    
    # Format 1: JSONL (Recommended for HF)
    with open(OUTPUT_HF_DATASET, "w", encoding="utf-8") as f:
        for example in hf_training_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"✓ JSONL format saved to '{OUTPUT_HF_DATASET}'")
    
    # Format 2: JSON array
    with open(OUTPUT_HF_DATASET_JSON, "w", encoding="utf-8") as f:
        json.dump(hf_training_data, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON array format saved to '{OUTPUT_HF_DATASET_JSON}'")

    # Format 3: Markdown documentation
    with open(OUTPUT_MD_FILE, "w", encoding="utf-8") as f:
        f.write("# Synthetic Legal Micro Cases and Deep Reasoning\n\n")
        f.write(f"**Total Cases Generated:** {len(successful_markdowns)}\n")
        f.write(f"**Total SFT Examples:** {len(hf_training_data)}\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        for case_id, md_text in successful_markdowns:
            f.write(f"\n\n=========================================\n")
            f.write(f"# Case ID: {case_id}\n")
            f.write(f"=========================================\n\n")
            f.write(md_text)
    print(f"✓ Markdown documentation saved to '{OUTPUT_MD_FILE}'")

    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Successfully processed {len(successful_markdowns)} full cases")
    print(f"Generated {len(hf_training_data)} SFT training conversational turns")
    
    if failed_cases:
        print(f"\nFailed cases skipped: {len(failed_cases)}")
        print(f"   (Check {LOG_FILE} for details)")