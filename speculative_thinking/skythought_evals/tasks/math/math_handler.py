import random
from typing import Any, Dict, List, Optional

from skythought_evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)

from ..base import TaskHandler


class MathTaskHandler(TaskHandler):
    def generate_prompt(self, problem):
        return self.task_config.templating_parameters["template"].format(**problem)

    def check_correctness(self, problem, generation):
        answer = strip_answer_string(problem[self.task_config.answer_key])
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)

    def update_results(self, problem, response):
        if not isinstance(response, str):
            response = response.outputs[0].text.strip()
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        curr_res = self.check_correctness(problem, generation=response)
        if curr_res:
            response_entry["correctness"] = True
            response_entry["reason"] = ""
        else:
            response_entry["correctness"] = False
            response_entry["reason"] = "Solution is incorrect."

        return response_entry

    def make_conversations(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem)
            conversations.append(
                self.make_conversation_from_contents(
                    [prompt_text],
                    system_prompt=system_prompt,
                    user_template=user_template,
                )
            )
        return conversations

    def process_remaining_data(self, train_data, results, args=None, temperatures=None):
        """
        返回需要继续处理的数据，支持采样级别的恢复。
        对于部分完成的问题，会继续完成剩余的采样。
        """
        remaining = []
        
        for _, row in train_data.iterrows():
            problem_key = str(row[self.question_key])
            
            if problem_key not in results:
                # 完全未开始的问题，需要处理
                remaining.append(row.to_dict())
            else:
                # 使用通用方法检查是否需要更多采样
                if self._check_needs_more_samples(results[problem_key], args, temperatures):
                    remaining.append(row.to_dict())
        
        return remaining

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None, args=None
    ):
        dataset = self.load_dataset(subset=subset, split=split)
        
        # Check if this is math500 dataset - if so, use spec_reason.py compatible sampling
        if self.task_config.dataset_path == "qq8933/MATH500":
            # Use the same logic as spec_reason.py for MATH-500 dataset
            random.seed(42)  # Fixed seed to match spec_reason.py
            indices = random.sample(range(len(dataset)), 100)
            dataset = dataset.select(indices)
            return dataset.to_pandas()
        
        # Default behavior: convert to pandas and use start/end slicing
        dataset = dataset.to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
