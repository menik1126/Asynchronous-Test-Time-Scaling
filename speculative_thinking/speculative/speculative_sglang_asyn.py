# import __init__
from speculative.spe_utils import *
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from utils.data_utils import read_yml
from utils.utils import *
from utils.qwen_math_parser import *
import sglang as sgl
from sglang.api import AsyncEngine
import asyncio
import time

def create_async_sglang_model(model_name, target_model_gpu, dtype='bfloat16', max_model_len=32768, base_gpu_id=0):
    class AsyncModelWorker:
        def __init__(self, model_name: str):
            print(f"Initializing async model '{model_name}' on GPU {base_gpu_id} (tp_size={target_model_gpu})")
            model_args = {
                "model_path": model_name,
                "tp_size": target_model_gpu,
                "dtype": dtype,
                "context_length": max_model_len,
                "device": "cuda",
                "trust_remote_code": True,
                "base_gpu_id": base_gpu_id
            }
            self.model = AsyncEngine(**model_args, skip_tokenizer_init=True)
            # 获取tokenizer用于text到token的转换
            self.tokenizer = self.model.tokenizer_manager.tokenizer
            print(f"Async model '{model_name}' successfully initialized on GPU {base_gpu_id}!")
        
        async def generate_async(self, generated_ids, sampling_params):
            # 异步生成
            outputs = await self.model.generate_async(
                input_ids=[generated_ids], 
                sampling_params=sampling_params
            )
            token_ids = outputs[0]['output_ids']
            return token_ids
    
    return AsyncModelWorker(model_name)

async def get_async_model_result(model, generated_ids, sampling_params):
    token_ids = await model.generate_async(generated_ids, sampling_params)
    return token_ids

class spe_thinking_sglang:
    def __init__(self, **config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['target_model_name'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
        max_model_len = config.get('max_model_len', 32768)
        dtype = config.get('dtype', 'bfloat16')
        
        # 获取GPU ID配置
        target_base_gpu_id = config.get('target_base_gpu_id', 0)
        speculative_base_gpu_id = config.get('speculative_base_gpu_id', 0)
        
        # 创建异步模型
        self.target_model = create_async_sglang_model(
            config['target_model_name'], 
            config['target_model_gpu'], 
            dtype=dtype,
            max_model_len=max_model_len,
            base_gpu_id=target_base_gpu_id
        )
        self.speculative_model = None
        if config['speculative_model_name'] is not None: 
            self.speculative_model = create_async_sglang_model(
                config['speculative_model_name'], 
                config['speculative_model_gpu'], 
                dtype=dtype,
                max_model_len=max_model_len,
                base_gpu_id=speculative_base_gpu_id
            )
        self.help_think_word_ids = None if config['help_think_word'] is None else self.tokenizer([config['help_think_word']], return_tensors="np",add_special_tokens=False)["input_ids"][0].tolist()
        self.help_recap_words_ids = self.tokenizer([config['help_recap_words']], return_tensors="np",add_special_tokens=False)["input_ids"][0].tolist()
        self.TRIGGER_TOKENS = config['TRIGGER_TOKENS']
        self.TARGET_VALIDATION_KEYWORDS = config['TARGET_VALIDATION_KEYWORDS']
        self.choose_large = config['choose_large']
        self.not_reasoning = config.get('not_reasoning', False)
        self.config = config

    def generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95, use_parallel=False):
        if self.speculative_model is None:
            return asyncio.run(self.normal_generate_async(messages, max_tokens, temperature, top_k, top_p))
        else:
            if use_parallel:
                return asyncio.run(self.speculative_generate_parallel_async(messages, max_tokens, temperature, top_k, top_p))
            else:
                return asyncio.run(self.speculative_generate_async(messages, max_tokens, temperature, top_k, top_p))

    def get_prompt_len(self,messages ):
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        return  len(generated_ids)
    
    def calculate_available_tokens(self,current_len, total_limit):
        return max(0, total_limit - current_len)

    async def normal_generate_async(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95):
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        prompt_len = len(generated_ids)
        # 计算可用的token数量（总限制减去prompt长度）
        available_tokens = max(0, max_tokens - prompt_len)
        sampling_params = {
            "max_new_tokens": available_tokens, 
            "temperature": temperature, 
            "top_k": top_k, 
            "top_p": top_p
        }
        spe_ids = await get_async_model_result(self.target_model, generated_ids, sampling_params)
        generated_text = self.tokenizer.decode(spe_ids, skip_special_tokens=True)
        num_tokens, correct_tokens, try_correct_num = len(spe_ids)-prompt_len, [], 0
        return generated_text, num_tokens, correct_tokens, try_correct_num

    async def speculative_generate_async(self, messages=None, max_tokens=100, temperature=0.6, top_k=50, top_p=0.95):
        start_time = time.time()  
        # 将字符串stop tokens转换为token IDs
        stop_token_ids = []
        for stop_token in self.TRIGGER_TOKENS:
            if isinstance(stop_token, str):
                token_ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
                stop_token_ids.extend(token_ids)
            else:
                stop_token_ids.append(stop_token)
        stop_token_ids.append(self.tokenizer.eos_token_id)
        
        # 计算当前可用的token数量（总限制减去当前已生成的token数）
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        prompt_len = len(generated_ids)
        sampling_params_one = {
            "max_new_tokens": self.calculate_available_tokens(prompt_len, 1024), 
            "temperature": temperature, 
            "top_k": top_k, 
            "top_p": top_p,
            "stop_token_ids": stop_token_ids
        }
        tgt_sampling_params_cache = {
            "max_new_tokens": self.calculate_available_tokens(prompt_len, self.config['max_target_tokens']), 
            "temperature": temperature, 
            "top_k": top_k, 
            "top_p": top_p
        }
        token_num, change_tokens, change_flag, begin = 0, 0, False, self.config['begin']
        negative_sent_num, recap_token_num = 0, self.config['original_recap_token_num']
        correct_tokens, try_correct_num = [], 0
        recap_after_negtive_num = self.config['recap_after_negative_num']
        while token_num <= max_tokens:
            if self.config['time_out'] is not None and self.config['time_out']>0:
                use_time = time.time() - start_time
                if use_time > self.config['time_out']: return None
            if not begin:
                # 动态更新sampling参数，确保不超过总限制
                current_len = len(generated_ids)
                sampling_params_one["max_new_tokens"] = self.calculate_available_tokens(current_len, 1024)
                one_token_id = await get_async_model_result(self.speculative_model, generated_ids, sampling_params_one)
                generated_ids.extend(one_token_id)
                # print(f"one_token_id: {one_token_id}")
                # print(f"one_token_id[-1]: {one_token_id[-1]}")
                # print(f"self.tokenizer.eos_token_id: {self.tokenizer.eos_token_id}")
                if one_token_id[-1] == self.tokenizer.eos_token_id : break
                one_token = self.tokenizer.decode(one_token_id[-5:], skip_special_tokens=True)
            if begin or any(trigger in one_token for trigger in self.TRIGGER_TOKENS): 
                if begin:
                    change_tokens = self.config['begin_token_num']
                    begin = False
                    change_flag = True
                    tgt_kv_candidate=None
                    spe_decoded_text = ''
                elif negative_sent_num >= recap_after_negtive_num:
                    generated_ids.extend(self.help_recap_words_ids)
                    change_tokens = recap_token_num
                    change_flag = True
                    negative_sent_num = 0
                    recap_token_num, recap_after_negtive_num= min(recap_token_num + self.config['add_each_recap'],self.config['max_recap_token_num']), min(recap_after_negtive_num+self.config['add_each_neg'], self.config['max_negative_num'])
                else:
                    if self.help_think_word_ids is not None:
                        generated_ids.extend(self.help_think_word_ids)
                    # 动态更新sampling参数，确保不超过总限制
                    current_len = len(generated_ids)
                    tgt_sampling_params_cache["max_new_tokens"] = self.calculate_available_tokens(current_len, self.config['max_target_tokens'])
                    # 并行启动推测模型和目标模型的推理
                    spe_future = get_async_model_result(self.speculative_model, generated_ids, tgt_sampling_params_cache)
                    
                    # 等待推测结果
                    spe_ids = await spe_future
                    spe_token = self.tokenizer.decode(spe_ids, skip_special_tokens=True)
                    spe_sent = sentiment_analysis(spe_token, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                    
                    if self.not_reasoning or spe_sent != 0:
                        try_correct_num = try_correct_num+1
                        # 动态更新sampling参数，确保不超过总限制
                        current_len = len(generated_ids)
                        tgt_sampling_params_cache["max_new_tokens"] = self.calculate_available_tokens(current_len, self.config['max_target_tokens'])
                        # 并行启动目标模型推理（不等待推测结果完成）
                        tgt_future = get_async_model_result(self.target_model, generated_ids, tgt_sampling_params_cache)
                        
                        # 等待目标模型结果
                        tgt_ids = await tgt_future
                        tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                        tgt_sent = sentiment_analysis(tgt_token, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                        
                        if self.choose_large or (spe_sent<0 and tgt_sent >=0) or (spe_sent>0 and tgt_sent<0):
                            decode_text = tgt_token
                            correct_tokens.append({
                                'pos': len(generated_ids)-prompt_len, 'token_num':self.config['max_target_tokens'],
                                'traget':tgt_token, 'speculative':spe_token})
                            generated_ids.extend(tgt_ids)
                            final_sent=tgt_sent
                        else:
                            generated_ids.extend(spe_ids)
                            decode_text = spe_token
                            final_sent=spe_sent
                        if final_sent < 0: negative_sent_num = negative_sent_num+1
                        if contains_keywords(decode_text, self.TARGET_VALIDATION_KEYWORDS['verify']):
                            change_tokens = self.config['original_recap_token_num']
                            change_flag = True
                if change_flag:
                    try_correct_num = try_correct_num+1
                    # 动态计算可用的token数量，取change_tokens和剩余可用token的较小值
                    current_len = len(generated_ids)
                    available_tokens = self.calculate_available_tokens(current_len, max_tokens)
                    actual_change_tokens = min(change_tokens, available_tokens)
                    tgt_sampling_params_= {
                        "max_new_tokens": actual_change_tokens, 
                        "temperature": temperature, 
                        "top_k": top_k, 
                        "top_p": top_p
                    } 
                    tgt_ids = await get_async_model_result(self.target_model, generated_ids, tgt_sampling_params_)    
                    tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                    correct_tokens.append({
                        'pos': len(generated_ids)-prompt_len, 'token_num':change_tokens,
                        'traget':tgt_token})
                    generated_ids.extend(tgt_ids)
                    change_flag = False
            token_num = len(generated_ids)
            if self.tokenizer.eos_token_id in generated_ids[-self.config['max_target_tokens']:]: 
                break
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, len(generated_ids)-prompt_len, correct_tokens, try_correct_num

    async def speculative_generate_parallel_async(self, messages=None, max_tokens=100, temperature=0.6, top_k=50, top_p=0.95):
        """
        高级并行推测推理版本，实现真正的异步流水线
        类似于vLLM版本的Ray异步模式
        """
        start_time = time.time()  
        
        # 将字符串stop tokens转换为token IDs
        stop_token_ids = []
        for stop_token in self.TRIGGER_TOKENS:
            if isinstance(stop_token, str):
                token_ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
                stop_token_ids.extend(token_ids)
            else:
                stop_token_ids.append(stop_token)
        stop_token_ids.append(self.tokenizer.eos_token_id)
        
        # 计算当前可用的token数量（总限制减去当前已生成的token数）
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        prompt_len = len(generated_ids)
        sampling_params_one = {
            "max_new_tokens": self.calculate_available_tokens(prompt_len, 1024), 
            "temperature": temperature, 
            "top_k": top_k, 
            "top_p": top_p,
            "stop_token_ids": stop_token_ids
        }
        tgt_sampling_params_cache = {
            "max_new_tokens": self.calculate_available_tokens(prompt_len, self.config['max_target_tokens']), 
            "temperature": temperature, 
            "top_k": top_k, 
            "top_p": top_p
        }
        
        token_num, change_tokens, change_flag, begin = 0, 0, False, self.config['begin']
        negative_sent_num, recap_token_num = 0, self.config['original_recap_token_num']
        correct_tokens, try_correct_num = [], 0
        recap_after_negtive_num = self.config['recap_after_negative_num']
        
        # 异步任务队列
        pending_tasks = []
        
        while token_num <= max_tokens:
            if self.config['time_out'] is not None and self.config['time_out']>0:
                use_time = time.time() - start_time
                if use_time > self.config['time_out']: return None
                
            if not begin:
                # 动态更新sampling参数，确保不超过总限制
                current_len = len(generated_ids)
                sampling_params_one["max_new_tokens"] = self.calculate_available_tokens(current_len, 1024)
                # 异步启动初始推理
                one_token_future = get_async_model_result(self.speculative_model, generated_ids, sampling_params_one)
                one_token_id = await one_token_future
                generated_ids.extend(one_token_id)
                
                if one_token_id[-1] == self.tokenizer.eos_token_id : break
                one_token = self.tokenizer.decode(one_token_id[-5:], skip_special_tokens=True)
                
            if begin or any(trigger in one_token for trigger in self.TRIGGER_TOKENS): 
                if begin:
                    change_tokens = self.config['begin_token_num']
                    begin = False
                    change_flag = True
                    tgt_kv_candidate=None
                    spe_decoded_text = ''
                elif negative_sent_num >= recap_after_negtive_num:
                    generated_ids.extend(self.help_recap_words_ids)
                    change_tokens = recap_token_num
                    change_flag = True
                    negative_sent_num = 0
                    recap_token_num, recap_after_negtive_num= min(recap_token_num + self.config['add_each_recap'],self.config['max_recap_token_num']), min(recap_after_negtive_num+self.config['add_each_neg'], self.config['max_negative_num'])
                else:
                    if self.help_think_word_ids is not None:
                        generated_ids.extend(self.help_think_word_ids)
                    # 动态更新sampling参数，确保不超过总限制
                    current_len = len(generated_ids)
                    tgt_sampling_params_cache["max_new_tokens"] = self.calculate_available_tokens(current_len, self.config['max_target_tokens'])
                    # 并行启动推测模型和目标模型的推理
                    spe_future = get_async_model_result(self.speculative_model, generated_ids, tgt_sampling_params_cache)
                    tgt_future = get_async_model_result(self.target_model, generated_ids, tgt_sampling_params_cache)
                    
                    # 等待两个模型的结果
                    spe_ids, tgt_ids = await asyncio.gather(spe_future, tgt_future)
                    
                    spe_token = self.tokenizer.decode(spe_ids, skip_special_tokens=True)
                    spe_sent = sentiment_analysis(spe_token, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                    
                    if self.not_reasoning or spe_sent != 0:
                        try_correct_num = try_correct_num+1
                        
                        tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                        tgt_sent = sentiment_analysis(tgt_token, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                        
                        if self.choose_large or (spe_sent<0 and tgt_sent >=0) or (spe_sent>0 and tgt_sent<0):
                            decode_text = tgt_token
                            correct_tokens.append({
                                'pos': len(generated_ids)-prompt_len, 'token_num':self.config['max_target_tokens'],
                                'traget':tgt_token, 'speculative':spe_token})
                            generated_ids.extend(tgt_ids)
                            final_sent=tgt_sent
                        else:
                            generated_ids.extend(spe_ids)
                            decode_text = spe_token
                            final_sent=spe_sent
                        if final_sent < 0: negative_sent_num = negative_sent_num+1
                        if contains_keywords(decode_text, self.TARGET_VALIDATION_KEYWORDS['verify']):
                            change_tokens = self.config['original_recap_token_num']
                            change_flag = True
                            
                if change_flag:
                    try_correct_num = try_correct_num+1
                    # 动态计算可用的token数量，取change_tokens和剩余可用token的较小值
                    current_len = len(generated_ids)
                    available_tokens = self.calculate_available_tokens(current_len, max_tokens)
                    actual_change_tokens = min(change_tokens, available_tokens)
                    tgt_sampling_params_= {
                        "max_new_tokens": actual_change_tokens, 
                        "temperature": temperature, 
                        "top_k": top_k, 
                        "top_p": top_p
                    } 
                    tgt_ids = await get_async_model_result(self.target_model, generated_ids, tgt_sampling_params_)    
                    tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                    correct_tokens.append({
                        'pos': len(generated_ids)-prompt_len, 'token_num':change_tokens,
                        'traget':tgt_token})
                    generated_ids.extend(tgt_ids)
                    change_flag = False
                    
            token_num = len(generated_ids)
            if self.tokenizer.eos_token_id in generated_ids[-self.config['max_target_tokens']:]: 
                break
                
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, len(generated_ids)-prompt_len, correct_tokens, try_correct_num
        


if __name__ == "__main__":
    yml_path = '/home/wxy320/ondemand/program/speculative_thinking/speculative/config/nromal/32B.yml'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_yml(yml_path)
    model = spe_thinking_sglang(**config)
    messages = []
    messages.append({
        "role": "user",
        "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + 'how to define the question?' + ' <think>\n'
    })

    # 测试普通异步模式
    print("Testing normal async mode...")
    result1 = model.generate(messages, 1024, use_parallel=False)
    print(f"Normal async result: {result1[0][:100]}...")
    
    # 测试并行异步模式
    print("Testing parallel async mode...")
    result2 = model.generate(messages, 1024, use_parallel=True)
    print(f"Parallel async result: {result2[0][:100]}...")
