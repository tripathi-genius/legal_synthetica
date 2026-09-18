#!/bin/bash
set -e

echo ""
echo "Starting Legal Data Processing Pipeline"
echo ""

echo -e "\n[1/3] Running Chunking and Taxonomy Classification..."
python3 chunking_and_taxonomy.py
echo "✓ Chunking and Taxonomy complete."

# Moving the file from the nfs directory to the local directory
#mv nfs/task_class_saved.pickle ./task_class_saved.pickle

echo -e "\n[2/3] Running Entity-Relationship Extraction..."
python3 er_extractor.py
echo "✓ ER Extraction complete."

if [ -f "extracted_chunks_raw.json" ]; then
    echo "Renaming extraction output to match Micro Generator expectations..."
    mv extracted_chunks_raw.json extracted_er_with_chunks.json
fi

echo -e "\n[3/3] Running Micro Case Generator..."
python3 micro_generator_gpt_oss.py
echo "✓ Micro Case Generation complete."

echo -e "\n========================================"
echo "Pipeline execution finished successfully!"
echo "========================================"
