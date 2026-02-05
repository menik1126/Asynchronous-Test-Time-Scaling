# import __init__
from speculative.spe_utils import *
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from utils.data_utils import read_yml
from utils.utils import *
from utils.qwen_math_parser import *
from vllm import LLM, SamplingParams
import ray
import time

def create_ray_model(model_name, target_model_gpu, dtype='bfloat16', max_model_len=32768):
    @ray.remote(num_gpus=target_model_gpu)
    class ModelWorkerSingleGPU:
        def __init__(self, model_name: str):
            print(f"[DEBUG] 初始化模型: {model_name}")
            print(f"[DEBUG] max_model_len: {max_model_len}")
            self.model = LLM(
                model=model_name, 
                tensor_parallel_size=target_model_gpu, 
                dtype=dtype,
                enable_prefix_caching=True, 
                max_model_len=max_model_len,
                gpu_memory_utilization=0.90,  # 降低内存使用率
                max_num_seqs=2,  # 减少并发序列数
                enforce_eager=True,  # 禁用CUDA图捕获
                max_num_batched_tokens=max_model_len  # 确保 >= max_model_len
            )
            print(f"[DEBUG] 模型 {model_name} 初始化成功")

                
        def generate(self, generated_ids, sampling_params):
            try:
                # 检查输入长度是否超过vLLM模型的最大上下文长度
                max_context_len = 8192  # vLLM模型的最大上下文长度
                if len(generated_ids) >= max_context_len:
                    print(f"[WARNING] 输入长度 {len(generated_ids)} 已达到或超过vLLM最大上下文长度 {max_context_len}")
                    # 截断输入到安全长度
                    safe_length = max_context_len - 100  # 保留100个token的安全边界
                    generated_ids = generated_ids[:safe_length]
                    print(f"[INFO] 截断输入到 {len(generated_ids)} tokens")
                
                outputs = self.model.generate(
                    prompt_token_ids=generated_ids, 
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
                token_ids = list(outputs[0].outputs[0].token_ids)
                return token_ids
            except Exception as e:
                print(f"[ERROR] 生成过程中出错: {e}")
                raise e
    return ModelWorkerSingleGPU.remote(model_name)

def get_ray_reuslt(model, generated_ids, sampling_params):
    result_a_future = model.generate.remote(generated_ids, sampling_params)
    token_ids = ray.get(result_a_future)
    return token_ids

class spe_thinking_vllm:
    def calculate_available_tokens(self, current_len, total_limit):
        return max(0, total_limit - current_len)
    def __init__(self, **config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['target_model_name'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
        self.max_model_len = config.get('max_model_len', 32768)
        dtype = config.get('dtype', 'bfloat16')
        self.target_model = create_ray_model(config['target_model_name'], config['target_model_gpu'], dtype=dtype, max_model_len=self.max_model_len)
        self.speculative_model = None
        if config['speculative_model_name'] is not None: 
            self.speculative_model = create_ray_model(config['speculative_model_name'], config['speculative_model_gpu'], dtype=dtype, max_model_len=self.max_model_len)
        self.help_think_word_ids = None if config['help_think_word'] is None else self.tokenizer([config['help_think_word']], return_tensors="np",add_special_tokens=False)["input_ids"][0].tolist()
        self.help_recap_words_ids = self.tokenizer([config['help_recap_words']], return_tensors="np",add_special_tokens=False)["input_ids"][0].tolist()
        self.TRIGGER_TOKENS = config['TRIGGER_TOKENS']
        self.TARGET_VALIDATION_KEYWORDS = config['TARGET_VALIDATION_KEYWORDS']
        self.choose_large = config['choose_large']
        self.not_reasoning = config.get('not_reasoning', False)
        self.config = config

    def generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95, return_only_generated=False):
        if self.speculative_model is None:
            return self.normal_generate( messages, max_tokens, temperature, top_k, top_p, return_only_generated)
        else:
            return self.speculative_generate( messages, max_tokens, temperature, top_k, top_p, return_only_generated)

    def get_prompt_len(self,messages ):
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        return  len(generated_ids)

    def normal_generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95, return_only_generated=False):
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        prompt_len = len(generated_ids)
        
        # 检查输入长度是否已经超过限制
        if prompt_len >= self.max_model_len:
            print(f"[WARNING] 输入prompt长度 {prompt_len} 已达到或超过最大长度限制 {self.max_model_len}")
            # 返回截断的prompt和相关信息
            truncated_ids = generated_ids[:self.max_model_len-100]  # 保留一些空间给生成
            generated_text = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
            if return_only_generated:
                return "", 0, [], 0
            return generated_text, 0, [], 0
        
        # 计算可用的token数量（总限制减去prompt长度）
        available_tokens = max(0, max_tokens - prompt_len)
        sampling_params = SamplingParams(max_tokens=available_tokens, temperature=temperature, top_k=top_k, top_p=top_p, 
                                            skip_special_tokens=False)
        spe_ids = get_ray_reuslt(self.target_model, generated_ids, sampling_params)
        
        # 检查返回结果是否为空
        if not spe_ids:
            print(f"[WARNING] target_model返回空结果，可能是长度超限")
            # 返回原始prompt
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text, 0, [], 0
            
        generated_text = self.tokenizer.decode(spe_ids, skip_special_tokens=True)
        num_tokens, correct_tokens, try_correct_num = len(spe_ids)-prompt_len, [], 0
        
        if return_only_generated:
            # 只返回生成的部分，不包括prompt
            generated_only_text = self.tokenizer.decode(spe_ids[prompt_len:], skip_special_tokens=True)
            return generated_only_text, num_tokens, correct_tokens, try_correct_num
        
        return generated_text, num_tokens, correct_tokens, try_correct_num

    def speculative_generate(self, messages=None, max_tokens=100, temperature=0.6, top_k=50, top_p=0.95, return_only_generated=False):
        start_time = time.time()  
        stops = self.TRIGGER_TOKENS+[self.tokenizer.eos_token ] 
        
        # 计算当前可用的token数量（总限制减去当前已生成的token数）
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        prompt_len = len(generated_ids)
        
        # 检查输入长度是否已经超过限制
        if prompt_len >= self.max_model_len:
            print(f"[WARNING] 输入prompt长度 {prompt_len} 已达到或超过最大长度限制 {self.max_model_len}")
            # 返回截断的prompt和相关信息
            truncated_ids = generated_ids[:self.max_model_len-100]  # 保留一些空间给生成
            generated_text = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
            if return_only_generated:
                # 只返回生成的部分（这里没有生成，所以返回空字符串）
                return "", 0, [], 0
            return generated_text, 0, [], 0
        
        # vLLM的max_tokens是总共要生成的token数量，不是新生成的token数量
        # 所以我们需要确保至少生成1个token
        max_tokens_one = max(1, min(1024, self.max_model_len - prompt_len))
        max_tokens_cache = max(1, min(self.config['max_target_tokens'], self.max_model_len - prompt_len))
        
        sampling_params_one= SamplingParams(max_tokens=max_tokens_one, temperature=temperature, top_k=top_k, top_p=top_p, 
                                            skip_special_tokens=False, stop=stops)
        tgt_sampling_params_cache= SamplingParams(max_tokens=max_tokens_cache, temperature=temperature, top_k=top_k, top_p=top_p,
                                                  skip_special_tokens=False)
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
                
                # 检查当前长度是否接近最大限制
                if current_len >= self.max_model_len - 100:  # 保留100个token的安全边界
                    print(f"[WARNING] 当前长度 {current_len} 接近最大限制 {self.max_model_len}，停止生成")
                    break
                    
                # vLLM的max_tokens是总共要生成的token数量
                max_tokens_one = max(1, min(1024, self.max_model_len - current_len))
                sampling_params_one.max_tokens = max_tokens_one
                one_token_id = get_ray_reuslt(self.speculative_model, generated_ids, sampling_params_one)
                
                # 检查返回结果是否为空
                if not one_token_id:
                    print(f"[WARNING] speculative_model返回空结果，可能是长度超限")
                    break
                    
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
                    
                    # 检查当前长度是否接近最大限制
                    if current_len >= self.max_model_len - 100:
                        print(f"[WARNING] 当前长度 {current_len} 接近最大限制 {self.max_model_len}，停止推测思考")
                        break
                        
                    # vLLM的max_tokens是总共要生成的token数量
                    max_tokens_cache = max(1, min(self.config['max_target_tokens'], self.max_model_len - current_len))
                    tgt_sampling_params_cache.max_tokens = max_tokens_cache
                    spe_ids = get_ray_reuslt(self.speculative_model, generated_ids, tgt_sampling_params_cache)
                    
                    # 检查返回结果是否为空
                    if not spe_ids:
                        print(f"[WARNING] speculative_model推测返回空结果")
                        break
                        
                    spe_token = self.tokenizer.decode(spe_ids, skip_special_tokens=True)
                    spe_sent = sentiment_analysis(spe_token, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                    if self.not_reasoning or spe_sent != 0:
                        try_correct_num = try_correct_num+1
                        
                        # 再次检查长度
                        current_len = len(generated_ids)
                        if current_len >= self.max_model_len - 100:
                            print(f"[WARNING] 当前长度 {current_len} 接近最大限制 {self.max_model_len}，停止纠正")
                            break
                            
                        # vLLM的max_tokens是总共要生成的token数量
                        max_tokens_cache = max(1, min(self.config['max_target_tokens'], self.max_model_len - current_len))
                        tgt_sampling_params_cache.max_tokens = max_tokens_cache
                        tgt_ids = get_ray_reuslt(self.target_model, generated_ids, tgt_sampling_params_cache)
                        
                        # 检查返回结果是否为空
                        if not tgt_ids:
                            print(f"[WARNING] target_model纠正返回空结果")
                            break
                            
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
                    
                    # 检查当前长度是否接近最大限制
                    if current_len >= self.max_model_len - 100:
                        print(f"[WARNING] 当前长度 {current_len} 接近最大限制 {self.max_model_len}，停止change操作")
                        break
                    
                    # vLLM的max_tokens是总共要生成的token数量
                    max_tokens_change = max(1, min(change_tokens, self.max_model_len - current_len))
                    
                    tgt_sampling_params_= SamplingParams(max_tokens=max_tokens_change, temperature=temperature, 
                                                         top_k=top_k, top_p=top_p, skip_special_tokens=False) 
                    tgt_ids = get_ray_reuslt(self.target_model, generated_ids, tgt_sampling_params_)
                    
                    # 检查返回结果是否为空
                    if not tgt_ids:
                        print(f"[WARNING] target_model change返回空结果")
                        break
                        
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
        
        if return_only_generated:
            # 只返回生成的部分，不包括prompt
            generated_only_text = self.tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)
            return generated_only_text, len(generated_ids)-prompt_len, correct_tokens, try_correct_num
        
        return generated_text, len(generated_ids)-prompt_len, correct_tokens, try_correct_num
        


if __name__ == "__main__":
    yml_path = '/home/wxy320/ondemand/program/speculative_thinking/speculative/config/nromal/32B.yml'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_yml(yml_path)
    model = spe_thinking_vllm(**config)
    messages = []
    messages.append({
        "role": "user",
        "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + 'how to define the question?' + ' <think>\n'
    })

    model.generate(messages, 1024)
