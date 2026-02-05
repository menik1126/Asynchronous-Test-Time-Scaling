OUTPUT_DIR=./results
python spec_reason.py --dataset_name aime --problem_id 60 --repeat_id 0 --score_threshold 7.0 --score_method greedy --token_budget 8192 --output_dir "$OUTPUT_DIR"