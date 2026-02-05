import os
import json
import argparse
import re
from tqdm import tqdm
from dataset import load_my_dataset
from async_agent import anyone_check
import asyncio

async def compute_accuracy_from_files(output_dir, dataset_name, repeats):
    print(f"Loading ground truth answers from dataset: {dataset_name}...")
    _, all_answers = load_my_dataset(dataset_name, repeats)
    total_problems = len(all_answers) // repeats
    
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' not found.")
        return

    result_files = [f for f in os.listdir(output_dir) if f.startswith("problem_") and f.endswith(".json")]
    
    result_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

    if not result_files:
        print("No result files found in the directory.")
        return

    generated_answers = []
    
    print("Extracting generated answers from result files...")
    for filename in tqdm(result_files, desc="Processing result files"):
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'final_answer' in data:
                    generated_answers.append(data['final_answer'])
                else:
                    print(f"Warning: 'final_answer' not found in {filename}.")
                    generated_answers.append("invalid")
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON from {filename}. Skipping.")
                generated_answers.append("invalid")

    right_count = 0
    
    if len(generated_answers) != len(all_answers):
        print(f"Warning: Number of generated answers ({len(generated_answers)}) does not match number of problems in dataset ({len(all_answers)}).")
        min_len = min(len(generated_answers), len(all_answers))
        generated_answers = generated_answers[:min_len]
        all_answers = all_answers[:min_len]

    group = len(generated_answers) // repeats
    for i in tqdm(range(group), desc="Comparing answers"):
        start = i * repeats
        end = (i + 1) * repeats
        
        outputs = generated_answers[start:end]
        correct_answer = all_answers[start]

        ans_status = await anyone_check(correct_answer, outputs)
        if ans_status == "Match":
            right_count += 1

    accuracy = right_count / group
    print(f"\nFinal Accuracy: {accuracy:.2%}")
    print(f"Correct: {right_count} / Total: {group}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute accuracy from saved evaluation results.")
    parser.add_argument("--output_dir", type=str, 
                        required=True,
                        help="The directory containing the JSON result files.")
    parser.add_argument("--dataset_name", type=str,
                        default="aime25",
                        help="Name of the dataset used (e.g., gpqa, math500).")
    parser.add_argument("--repeats", type=int, default=16,
                        help="Number of times each problem was repeated in the original run.")
    
    args = parser.parse_args()

    asyncio.run(compute_accuracy_from_files(args.output_dir, args.dataset_name, args.repeats))