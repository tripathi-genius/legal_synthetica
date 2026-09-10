Here is I deployed models using VLLM. I will provide pre-trained models in the repository so that experiment can be reproduced. 


python llm_judge.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_llama-8b-base_20260907_175712.jsonl --output_dir llama8b_base_eval 

python llm_judge.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_qwen-3b-base_20260907_165829.jsonl --output_dir qwen3b_base_eval 


python llm_judge.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_llama-3b-finetuned_20260907_192207.jsonl --output_dir llama3b_finetune_eval 


python llm_judge.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_llama-8b-finetuned_20260907_202421.jsonl --output_dir llama8b_finetuned_eval 


python llm_judge.py --reference test_micro_cases/sft_test_final.jsonl --prediction inference_qwen-3b-finetuned_20260907_193946.jsonl --output_dir qwen3b_finetuned_eval 


