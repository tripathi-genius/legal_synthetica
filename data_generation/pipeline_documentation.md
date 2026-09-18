# Legal Data Processing Pipeline

This repository contains the data processing pipeline for extracting taxonomy, mapping logical entities and relationships, and generating synthetic legal micro-cases. 

## Pipeline Stages

The pipeline consists of three main sequential scripts:

1. **Chunking & Taxonomy Classification (`chunking_and_taxonomy.py`)**
   - **Action:** Loads the raw dataset, splits the judgment text into manageable chunks (512-1024 characters), and queries an LLM to extract the document's taxonomy and key topics.
   - **Output:** Saves the chunked and classified data to a pickle file (`task_class_saved.pickle`).

2. **Entity-Relationship (ER) Extraction (`er_extractor.py`)**
   - **Action:** Ingests the chunked data and uses a pool of local and remote LLMs to extract key entities (Facts, Claims, Precedents, Acts) and map the logical relationships between them (e.g., "proves", "contradicts").
   - **Output:** Saves the structured logic maps to a JSON file (`extracted_chunks_raw.json`).

3. **Micro Case Generation (`micro_generator_gpt_oss.py`)**
   - **Action:** Groups the extracted logical chunks back by case ID. It sequentially builds a cohesive factual narrative ("Micro Case") and then generates a deep reasoning framework based on those facts.
   - **Output:** Generates synthetic conversational turns for Supervised Fine-Tuning (SFT) and outputs markdown documentation (`micro_cases_and_args.md`).

## Training vs. Testing Data Generation

**IMPORTANT:** This exact same pipeline is used to generate *both* the training and testing datasets. 

*Note: We are going to change the input data source (train split vs. test split) and potentially adjust the LLM endpoints/temperatures depending on whether we are generating the training dataset or the testing dataset to ensure strict data separation and appropriate generation parameters.*