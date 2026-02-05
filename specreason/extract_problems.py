#!/usr/bin/env python3
"""
抽取所有题目的脚本
从Qwen-32B_deepseek-1.5B目录下的每个文件夹中提取pickle文件中的题目信息
"""

import pickle
import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_pickle_data(pickle_path: str) -> Dict[str, Any]:
    """加载pickle文件数据"""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return None

def extract_final_answer(reasoning_steps: List[Dict]) -> Optional[str]:
    """
    从推理步骤中提取最终答案
    优先查找\\boxed{}格式，否则提取最后一步中的数字
    """
    if not reasoning_steps:
        return None
    
    # 获取最后一步
    last_step = reasoning_steps[-1].get('step_str', '')
    
    # 1. 首先查找 \boxed{} 格式的答案
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, last_step)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 2. 如果没有 \boxed，查找最后几步中的答案指示词
    answer_patterns = [
        r'(?:答案是|answer is|答案为|the answer is)\s*([0-9]+)',
        r'(?:因此|therefore|所以|thus)[^0-9]*([0-9]+)',
        r'(?:最终|final|结果|result)[^0-9]*([0-9]+)'
    ]
    
    # 检查最后几步
    for step in reversed(reasoning_steps[-3:]):
        step_text = step.get('step_str', '')
        for pattern in answer_patterns:
            match = re.search(pattern, step_text, re.IGNORECASE)
            if match:
                return match.group(1)
    
    # 3. 如果都没找到，提取最后一步中的最后一个数字
    numbers = re.findall(r'\b\d+\b', last_step)
    if numbers:
        return numbers[-1]
    
    # 4. 如果最后一步没有数字，检查最后几步
    for step in reversed(reasoning_steps[-3:]):
        step_text = step.get('step_str', '')
        numbers = re.findall(r'\b\d+\b', step_text)
        if numbers:
            return numbers[-1]
    
    return None

def extract_all_problems(base_dir: str, k: int = 16, num_problems: int = 15) -> List[Dict[str, Any]]:
    """
    从指定目录下抽取指定数量的题目，计算Pass@k
    
    Args:
        base_dir: Qwen-32B_deepseek-1.5B目录的路径
        k: Pass@k中的k值，默认为16
        num_problems: 要处理的题目数量，默认为15
    
    Returns:
        包含指定数量题目信息的列表，每道题包含多次解题结果
    """
    problems = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"目录不存在: {base_dir}")
        return problems
    
    # 遍历所有数字命名的文件夹，只取指定数量
    folders = sorted([f for f in base_path.iterdir() if f.is_dir() and f.name.isdigit()])[:num_problems]
    
    for folder in folders:
        print(f"正在处理题目 {folder.name}...")
        
        # 获取所有pickle文件
        pickle_files = sorted(folder.glob("*.pickle"))
        
        if not pickle_files:
            print(f"题目 {folder.name} 缺少pickle文件")
            continue
        
        # 存储这道题的所有解题结果
        problem_attempts = []
        
        for pickle_file in pickle_files:
            data = load_pickle_data(str(pickle_file))
            
            if data:
                # 提取最终答案
                final_answer = extract_final_answer(data.get('reasoning_steps', []))
                
                # 判断答案是否正确
                ground_truth = str(data.get('ground_truth', '')).strip()
                is_correct = ground_truth == str(final_answer).strip() if final_answer else False
                
                attempt_info = {
                    'attempt_id': pickle_file.stem,  # 文件名（如 "0", "1", "2"）
                    'final_answer': final_answer,
                    'is_correct': is_correct,
                    'total_steps': data.get('total_steps'),
                    'total_tokens': data.get('total_tokens'),
                    'reasoning_steps': data.get('reasoning_steps')
                }
                problem_attempts.append(attempt_info)
        
        if problem_attempts:
            # 获取第一份数据作为题目基本信息
            first_data = load_pickle_data(str(pickle_files[0]))
            
            # 计算Pass@k统计
            total_attempts = len(problem_attempts)
            correct_attempts = sum(1 for attempt in problem_attempts if attempt['is_correct'])
            
            # 计算Pass@k
            pass_at_k = 1.0 if correct_attempts > 0 else 0.0
            
            # 打印中间信息
            print(f"  题目ID: {first_data.get('problem_id')}")
            print(f"  数据集: {first_data.get('dataset_name')}")
            print(f"  标准答案: {first_data.get('ground_truth')}")
            print(f"  解题次数: {total_attempts}")
            print(f"  正确次数: {correct_attempts}")
            print(f"  Pass@{k}: {pass_at_k}")
            print(f"  文件夹: {folder.name}")
            print("-" * 60)
            
            # 提取关键信息
            problem_info = {
                'problem_id': first_data.get('problem_id'),
                'dataset_name': first_data.get('dataset_name'),
                'problem_text': first_data.get('problem_text'),
                'ground_truth': first_data.get('ground_truth'),
                'options': first_data.get('options'),
                'folder_name': folder.name,
                'total_attempts': total_attempts,
                'correct_attempts': correct_attempts,
                'pass_at_k': pass_at_k,
                'k_value': k,
                'attempts': problem_attempts
            }
            problems.append(problem_info)
        else:
            print(f"无法加载题目 {folder.name} 的数据")
    
    return problems

def save_problems_to_json(problems: List[Dict[str, Any]], output_file: str):
    """将题目保存为JSON文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)
        print(f"成功保存 {len(problems)} 道题目到 {output_file}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def save_problems_summary(problems: List[Dict[str, Any]], output_file: str, k: int = 16):
    """保存题目摘要信息，包含Pass@k统计"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"题目摘要 (Pass@{k}统计)\n")
            f.write("="*70 + "\n\n")
            
            for i, problem in enumerate(problems, 1):
                f.write(f"题目 {i} (ID: {problem.get('problem_id', 'N/A')}, 文件夹: {problem.get('folder_name', 'N/A')})\n")
                f.write(f"数据集: {problem.get('dataset_name', 'N/A')}\n")
                f.write(f"题目内容预览: {problem.get('problem_text', '')[:200]}...\n")
                f.write(f"标准答案: {problem.get('ground_truth', 'N/A')}\n")
                f.write(f"解题次数: {problem.get('total_attempts', 'N/A')}\n")
                f.write(f"正确次数: {problem.get('correct_attempts', 'N/A')}\n")
                f.write(f"Pass@{k}: {problem.get('pass_at_k', 'N/A')}\n")
                
                # 显示每次解题的简要信息
                attempts = problem.get('attempts', [])
                f.write("解题详情:\n")
                for attempt in attempts:
                    status = "✓" if attempt['is_correct'] else "✗"
                    f.write(f"  尝试{attempt['attempt_id']}: {attempt['final_answer']} {status}\n")
                
                f.write("-" * 70 + "\n\n")
            
            # 添加总体统计摘要
            f.write("=" * 80 + "\n")
            f.write(f"总体统计摘要 (Pass@{k})\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本统计
            f.write(f"总题目数: {len(problems)}\n")
            
            datasets = set(p.get('dataset_name') for p in problems)
            f.write(f"数据集: {', '.join(datasets)}\n")
            
            # 统计步骤和令牌
            all_attempts = []
            for problem in problems:
                all_attempts.extend(problem.get('attempts', []))
            
            valid_steps = [attempt.get('total_steps', 0) for attempt in all_attempts if attempt.get('total_steps')]
            valid_tokens = [attempt.get('total_tokens', 0) for attempt in all_attempts if attempt.get('total_tokens')]
            
            total_steps = sum(valid_steps)
            total_tokens = sum(valid_tokens)
            total_attempts = len(all_attempts)
            
            f.write(f"总解题次数: {total_attempts}\n")
            f.write(f"总步骤数: {total_steps}\n")
            f.write(f"平均步骤数: {total_steps / len(valid_steps):.2f}\n" if valid_steps else "平均步骤数: N/A\n")
            
            f.write(f"总令牌数: {total_tokens:,}\n")
            f.write(f"平均令牌数: {total_tokens / len(valid_tokens):.2f}\n" if valid_tokens else "平均令牌数: N/A\n")
            
            # 显示题目ID范围
            problem_ids = [p.get('problem_id') for p in problems if p.get('problem_id') is not None]
            if problem_ids:
                f.write(f"题目ID范围: {min(problem_ids)} - {max(problem_ids)}\n")
            
            # 计算总体Pass@k统计
            pass_at_k_count = sum(1 for problem in problems if problem.get('pass_at_k', 0) == 1.0)
            if len(problems) > 0:
                pass_at_k_rate = (pass_at_k_count / len(problems)) * 100
                f.write(f"\nPass@{k}统计:\n")
                f.write(f"  Pass@{k}: {pass_at_k_count}/{len(problems)} = {pass_at_k_rate:.1f}%\n")
            else:
                f.write(f"\nPass@{k}统计: N/A\n")
            
            # 计算总体准确率（传统方式）
            total_correct_attempts = sum(p.get('correct_attempts', 0) for p in problems)
            total_attempts = sum(p.get('total_attempts', 0) for p in problems)
            
            f.write(f"\n传统准确率统计:\n")
            f.write(f"总正确解题次数: {total_correct_attempts}\n")
            f.write(f"总解题次数: {total_attempts}\n")
            if total_attempts > 0:
                overall_accuracy = (total_correct_attempts / total_attempts) * 100
                f.write(f"总体准确率: {total_correct_attempts}/{total_attempts} = {overall_accuracy:.1f}%\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"成功保存题目摘要到 {output_file}")
    except Exception as e:
        print(f"保存摘要文件时出错: {e}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='题目抽取工具，支持Pass@k计算')
    parser.add_argument('--k', type=int, default=16, help='Pass@k中的k值 (默认: 16)')
    parser.add_argument('--num_problems', type=int, default=15, help='要处理的题目数量 (默认: 15)')
    parser.add_argument('--base_dir', type=str, 
                       default="/home/xiongjing/qj/specreason/results/sglang/greedy_3_sample_16/aime2025/Skywork/Skywork-OR1-32B_Skywork/Skywork-OR1-7B/",
                       help='目标目录路径')
    parser.add_argument('--output_dir', type=str, 
                       default="/home/xiongjing/qj/specreason/results/sglang/greedy_3_sample_16/aime2025/Skywork/Skywork-OR1-32B_Skywork/Skywork-OR1-7B/",
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    # 设置路径
    base_dir = args.base_dir
    output_dir = args.output_dir
    output_json = os.path.join(output_dir, f"extracted_problems_pass_at_{args.k}.json")
    output_summary = os.path.join(output_dir, f"problems_summary_pass_at_{args.k}.txt")
    
    print("=" * 70)
    print(f"题目抽取工具 (Pass@{args.k}计算)")
    print("=" * 70)
    print(f"目标目录: {base_dir}")
    print(f"处理题目数: {args.num_problems}")
    print(f"Pass@k值: {args.k}")
    print(f"输出JSON文件: {output_json}")
    print(f"输出摘要文件: {output_summary}")
    print("-" * 70)
    
    # 抽取指定数量的题目
    print(f"开始抽取前{args.num_problems}道题目...")
    problems = extract_all_problems(base_dir, args.k, args.num_problems)
    
    if problems:
        print("\n" + "=" * 70)
        print(f"抽取完成！成功处理了 {len(problems)} 道题目")
        print("=" * 70)
        
        # 保存为JSON文件
        print("\n正在保存文件...")
        save_problems_to_json(problems, output_json)
        
        # 保存摘要文件
        save_problems_summary(problems, output_summary, args.k)
        
        # 打印详细统计信息
        print(f"\n" + "=" * 60)
        print(f"详细统计信息 (Pass@{args.k})")
        print("=" * 60)
        print(f"总题目数: {len(problems)}")
        
        datasets = set(p.get('dataset_name') for p in problems)
        print(f"数据集: {', '.join(datasets)}")
        
        # 统计所有解题次数
        total_attempts = sum(p.get('total_attempts', 0) for p in problems)
        total_correct_attempts = sum(p.get('correct_attempts', 0) for p in problems)
        
        print(f"总解题次数: {total_attempts}")
        print(f"总正确解题次数: {total_correct_attempts}")
        
        # 统计步骤和令牌
        all_attempts = []
        for problem in problems:
            all_attempts.extend(problem.get('attempts', []))
        
        valid_steps = [attempt.get('total_steps', 0) for attempt in all_attempts if attempt.get('total_steps')]
        valid_tokens = [attempt.get('total_tokens', 0) for attempt in all_attempts if attempt.get('total_tokens')]
        
        total_steps = sum(valid_steps)
        total_tokens = sum(valid_tokens)
        
        print(f"总步骤数: {total_steps}")
        print(f"平均步骤数: {total_steps / len(valid_steps):.2f}" if valid_steps else "平均步骤数: N/A")
        
        print(f"总令牌数: {total_tokens:,}")
        print(f"平均令牌数: {total_tokens / len(valid_tokens):.2f}" if valid_tokens else "平均令牌数: N/A")
        
        # 显示题目ID范围
        problem_ids = [p.get('problem_id') for p in problems if p.get('problem_id') is not None]
        if problem_ids:
            print(f"题目ID范围: {min(problem_ids)} - {max(problem_ids)}")
        
        # 计算Pass@k统计
        pass_at_k_count = sum(1 for problem in problems if problem.get('pass_at_k', 0) == 1.0)
        if len(problems) > 0:
            pass_at_k_rate = (pass_at_k_count / len(problems)) * 100
            print(f"\nPass@{args.k}统计:")
            print(f"  Pass@{args.k}: {pass_at_k_count}/{len(problems)} = {pass_at_k_rate:.1f}%")
        else:
            print(f"\nPass@{args.k}统计: N/A")
        
        # 计算传统准确率
        if total_attempts > 0:
            overall_accuracy = (total_correct_attempts / total_attempts) * 100
            print(f"\n传统准确率: {total_correct_attempts}/{total_attempts} = {overall_accuracy:.1f}%")
        
        print("=" * 60)
        print("文件输出:")
        print(f"- JSON数据文件: {output_json}")
        print(f"- 题目摘要文件: {output_summary}")
        print("=" * 60)
        
    else:
        print("\n❌ 没有找到任何题目")
        print("请检查目标目录是否正确，以及是否包含有效的pickle文件")

if __name__ == "__main__":
    main()