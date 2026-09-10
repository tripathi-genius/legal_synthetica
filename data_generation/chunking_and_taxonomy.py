import re
import os
import json
import pickle
import requests
import threading
import itertools
import concurrent.futures
from openai import OpenAI
from datasets import load_dataset
from tqdm.auto import tqdm


TAXONOMY_PROMPT = """You are a legal taxonomy classification agent.
Analyze the provided document chunk and extract its taxonomy.
You MUST respond strictly with a JSON object enclosed in a ```json``` markdown code block.
Do not include any text outside the JSON block. If a field cannot be determined, use the string "UNKNOWN1".
Ensure the output is perfectly valid JSON (escape all internal quotes).
DO NOT use LaTeX, markdown text formatting, or unescaped backslashes inside the JSON strings.
Think before you answer.
Recheck the json structure before your answer and only output the correct output.

Please follow the following schema strictly:
{
    "taxonomy": {
        "document_type": "e.g. Ruling, laws, policies, etc.",
        "primary_domain": "The broad categorization of document e.g. appeal with revenue, criminal, civil, family, etc.",
        "section_purpose": "What this specific chunk is doing? (Background, fact finding, citations, precedents, etc.)"
    },
    "key_topics": ["List major key topics like 'a', 'b', 'c', ..."]
}
Do nothing else.
"""

BASE_URL = "http://api.llm.apps.os.dcs.gla.ac.uk/v1"
API_KEY = os.environ['IDA_LLM_API_KEY']
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


ENDPOINT_CONFIGS = [
    {"type": "openai", "model": "ministral-3-8b-reasoning-2512", "weight": 1},
    {"type": "openai", "model": "deepseek-r1-distil-qwen-14b", "weight": 1},
    {"type": "openai", "model": "ministral-3-14b-instruct-2512", "weight": 1},
    {"type": "local_vllm", "url": "http://localhost:8100/generate", "model": "local-llama-3.1-8b-fp8", "weight": 5}
]


ENDPOINTS = []
for config in ENDPOINT_CONFIGS:
    ENDPOINTS.extend([config] * config["weight"])


class ThreadSafeCycle:
    def __init__(self, iterable):
        self.iterable = itertools.cycle(iterable)
        self.lock = threading.Lock()
        
    def next(self):
        with self.lock:
            return next(self.iterable)

endpoint_cycle = ThreadSafeCycle(ENDPOINTS)


def format_llama3_prompt(system_msg, user_msg):
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def chunk_text(raw_text, min_chunk_length=512, max_chunk_length=1024):
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", raw_text) if p.strip()]

    processed_paragraphs = []
    for para in paragraphs:
        if len(para) <= max_chunk_length:
            processed_paragraphs.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            temp = ""
            for sentence in sentences:
                candidate = temp + " " + sentence if temp else sentence
                if len(candidate) <= max_chunk_length:
                    temp = candidate
                else:
                    if temp: processed_paragraphs.append(temp)
                    temp = sentence
            if temp: processed_paragraphs.append(temp)

    chunks = []
    current_chunk = ""
    for para in processed_paragraphs:
        candidate = current_chunk + "\n\n" + para if current_chunk else para
        if len(candidate) <= max_chunk_length:
            current_chunk = candidate
        else:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = para
            
    if current_chunk: chunks.append(current_chunk)
    return chunks

def prepare_chunks(data):
    data_id = data["case_id"]
    chunks = chunk_text(data["judgment_text"], min_chunk_length=512, max_chunk_length=1024)
    return [(data_id, i + 1, chunk) for i, chunk in enumerate(chunks)]

def parse_json_output(response_text):
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    json_block = match.group(1) if match else response_text.strip()
    
    cleaned_lines = []
    for line in json_block.split('\n'):
        if len(re.findall(r'(?<!\\)"', line)) % 2 != 0:
            stripped_line = line.rstrip()
            line = stripped_line[:-1] + '",' if stripped_line.endswith(',') else stripped_line + '"'
        cleaned_lines.append(line)
        
    json_block = '\n'.join(cleaned_lines)
    json_block = re.sub(r',\s*([\]}])', r'\1', json_block)
    json_block = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', json_block)
    
    try:
        return json.loads(json_block, strict=False)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}\nFailed block:\n{json_block}")
        return None


def get_llm_response(text, endpoint):
    if endpoint["type"] == "openai":
        messages = [
            {'role': 'system', 'content': TAXONOMY_PROMPT}, 
            {'role': 'user', 'content': text}
        ]
        
        result = client.chat.completions.create(
            model=endpoint["model"], 
            messages=messages,
            temperature=0.0
        )
        return result.choices[0].message.content
        
    elif endpoint["type"] == "local_vllm":
        prompt = format_llama3_prompt(TAXONOMY_PROMPT, text)
        payload = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 1024,
            "stream": False
        }
        
        response = requests.post(endpoint["url"], json=payload)
        response.raise_for_status()
        return response.json().get("text", "")

def process_chunk(args):
    data_id, chunk_id, chunk = args
    endpoint = endpoint_cycle.next()

    try:
        response_text = get_llm_response(chunk, endpoint)
        taxonomy_data = parse_json_output(response_text)
        
        if taxonomy_data is None:
            raise ValueError("Failed to extract valid JSON.")

        taxonomy_data.update({
            "text": chunk,
            "id": data_id,
            "chunk_id": chunk_id,
            "processed_by_model": endpoint["model"]
        })
        return taxonomy_data

    except Exception as e:
        return {
            "id": data_id,
            "chunk_id": chunk_id,
            "text": chunk,
            "error": str(e),
            "processed_by_model": endpoint["model"]
        }


if __name__ == "__main__":
    
    combined = load_dataset("mborcin/firac-appeals", "combined")
    
    tasks = []
    for data in tqdm(combined["train"], desc="Preparing chunks"):
        tasks.extend(prepare_chunks(data))
        
    print(f"Total chunks to process: {len(tasks)}")

    MAX_WORKERS = 80
    print(f"Executing with {MAX_WORKERS} concurrent threads using weighted routing...")

    data_chunked = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chunk, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Extracting taxonomy"):
            data_chunked.append(future.result())

    with open("nfs/task_class_saved.pickle", "wb") as file:
        pickle.dump(data_chunked, file)
        
    print("Processing complete. Saved to task_class_saved.pickle")