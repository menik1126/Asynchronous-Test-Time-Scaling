# import __init__
from speculative.spe_utils import *
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from utils.data_utils import read_yml
from utils.utils import *
from utils.qwen_math_parser import *
import sglang as sgl
#import sglang.api as sgl
import time

def create_sglang_model(model_name, target_model_gpu, dtype='bfloat16', max_model_len=32768, base_gpu_id=0):
    class ModelWorker:
        def __init__(self, model_name: str):
            print(f"Initializing model '{model_name}' on GPU {base_gpu_id} (tp_size={target_model_gpu})")
            model_args = {
                "model_path": model_name,
                "tp_size": target_model_gpu,
                "dtype": dtype,
                "context_length": max_model_len,
                "device": "cuda",
                "trust_remote_code": True,
                "base_gpu_id": base_gpu_id
            }
            self.model = sgl.Engine(**model_args, skip_tokenizer_init=True)
            # 获取tokenizer用于text到token的转换
            self.tokenizer = self.model.tokenizer_manager.tokenizer
            print(f"Model '{model_name}' successfully initialized on GPU {base_gpu_id}!")
        def generate(self, generated_ids, sampling_params):
            # 确保sampling_params包含返回token_ids的设置
            # print(f"[DEBUG] ModelWorker.generate调用")
            # print(f"[DEBUG] generated_ids长度: {len(generated_ids)}")
            # print(f"[DEBUG] sampling_params: {sampling_params}")
            
            try:
                # 检查输入长度是否超过sglang模型的最大上下文长度
                max_context_len = 8192  # sglang模型的最大上下文长度
                if len(generated_ids) >= max_context_len:
                    print(f"[WARNING] 输入长度 {len(generated_ids)} 已达到或超过sglang最大上下文长度 {max_context_len}")
                    # 截断输入到安全长度
                    safe_length = max_context_len - 100  # 保留100个token的安全边界
                    generated_ids = generated_ids[:safe_length]
                    print(f"[INFO] 截断输入到 {len(generated_ids)} tokens")
                
                outputs = self.model.generate(
                    input_ids=[generated_ids], 
                    sampling_params=sampling_params
                )
                # print(f"[DEBUG] model.generate成功，outputs类型: {type(outputs)}")
                token_ids = outputs[0]['output_ids']
                # print(f"[DEBUG] 提取的token_ids长度: {len(token_ids)}")
                return token_ids
            except Exception as e:
                # print(f"[DEBUG] ModelWorker.generate出错: {e}")
                print(f"[ERROR] ModelWorker.generate异常: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    return ModelWorker(model_name)

def get_model_result(model, generated_ids, sampling_params):
    # print(f"[DEBUG] get_model_result调用，sampling_params: {sampling_params}
        
        token_ids = model.generate(generated_ids, sampling_params)
        # print(f"[DEBUG] model.generate成功，返回token_ids长度: {len(token_ids) if token_ids else 0}")
        
        # 如果返回None，返回空列表
        if token_ids is None:
            print(f"[WARNING] model.generate返回None，可能是输入长度超限或其他错误")
            return []
            
        return token_ids


class spe_thinking_sglang:
    def __init__(self, **config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['target_model_name'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
        self.max_model_len = config.get('max_model_len', 32768)
        dtype = config.get('dtype', 'bfloat16')
        
        # 获取GPU ID配置
        target_base_gpu_id = config.get('target_base_gpu_id', 0)
        speculative_base_gpu_id = config.get('speculative_base_gpu_id', 0)
        
        self.target_model = create_sglang_model(
            config['target_model_name'], 
            config['target_model_gpu'], 
            dtype=dtype,
            max_model_len=self.max_model_len,
            base_gpu_id=target_base_gpu_id
        )
        self.speculative_model = None
        if config['speculative_model_name'] is not None: 
            self.speculative_model = create_sglang_model(
                config['speculative_model_name'], 
                config['speculative_model_gpu'], 
                dtype=dtype,
                max_model_len=self.max_model_len,
                base_gpu_id=speculative_base_gpu_id
            )
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
        sampling_params = {
            "max_new_tokens": available_tokens, 
            "temperature": temperature, 
            "top_k": top_k, 
            "top_p": top_p
        }
        spe_ids = get_model_result(self.target_model, generated_ids, sampling_params)
        
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
    def calculate_available_tokens(self,current_len, total_limit):
        return max(0, total_limit - current_len)

    def speculative_generate(self, messages=None, max_tokens=100, temperature=0.6, top_k=50, top_p=0.95, return_only_generated=False):
        # print(f"[DEBUG] 进入speculative_generate，max_tokens={max_tokens}, temperature={temperature}")
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
        # print(f"[DEBUG] stop_token_ids: {stop_token_ids}")
        
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
        
        # print(f"[DEBUG] prompt_len: {prompt_len}, max_tokens: {max_tokens}")
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
        # print(f"[DEBUG] sampling_params_one: {sampling_params_one}")
        # print(f"[DEBUG] tgt_sampling_params_cache: {tgt_sampling_params_cache}")
        token_num, change_tokens, change_flag, begin = 0, 0, False, self.config['begin']
        negative_sent_num, recap_token_num = 0, self.config['original_recap_token_num']
        # print(f"[DEBUG] 初始状态: begin={begin}, token_num={token_num}")
        
        
        correct_tokens, try_correct_num = [], 0
        recap_after_negtive_num = self.config['recap_after_negative_num']
        while token_num <= max_tokens:
            if self.config['time_out'] is not None and self.config['time_out']>0:
                use_time = time.time() - start_time
                # print(f"[DEBUG] 检查超时: use_time={use_time:.2f}s, time_out={self.config['time_out']}s")
                if use_time > self.config['time_out']: 
                    # print(f"[DEBUG] 超时返回None")
                    return None
            if not begin:
                # print(f"[DEBUG] 开始阶段，使用speculative_model生成")
                # 动态更新sampling参数，确保不超过总限制
                current_len = len(generated_ids)
                
                # 检查当前长度是否接近最大限制
                if current_len >= self.max_model_len - 100:  # 保留100个token的安全边界
                    print(f"[WARNING] 当前长度 {current_len} 接近最大限制 {self.max_model_len}，停止生成")
                    break
                    
                sampling_params_one["max_new_tokens"] = self.calculate_available_tokens(current_len, 1024)
                # print(f"[DEBUG] 调用speculative_model.generate，current_len={current_len}")
                one_token_id = get_model_result(self.speculative_model, generated_ids, sampling_params_one)
                # print(f"[DEBUG] speculative_model返回: {one_token_id}")
                
                # 检查返回结果是否为空
                if not one_token_id:
                    print(f"[WARNING] speculative_model返回空结果，可能是长度超限")
                    break
                    
                generated_ids.extend(one_token_id)
                # print(f"[DEBUG] one_token_id[-1]: {one_token_id[-1]}, eos_token_id: {self.tokenizer.eos_token_id}")
                if one_token_id[-1] == self.tokenizer.eos_token_id : 
                    # print(f"[DEBUG] 遇到EOS token，跳出循环")
                    break
                one_token = self.tokenizer.decode(one_token_id[-5:], skip_special_tokens=True)
                # print(f"[DEBUG] 解码的token: {one_token}")
            if begin or any(trigger in one_token for trigger in self.TRIGGER_TOKENS): 
                # print(f"[DEBUG] 触发推测思考条件: begin={begin}, one_token={one_token}")
                # print(f"[DEBUG] TRIGGER_TOKENS: {self.TRIGGER_TOKENS}")
                if begin:
                    # print(f"[DEBUG] 开始模式")
                    change_tokens = self.config['begin_token_num']
                    begin = False
                    change_flag = True
                    tgt_kv_candidate=None
                    spe_decoded_text = ''
                elif negative_sent_num >= recap_after_negtive_num:
                    # print(f"[DEBUG] 负面情绪过多，进行recap")
                    generated_ids.extend(self.help_recap_words_ids)
                    change_tokens = recap_token_num
                    change_flag = True
                    negative_sent_num = 0
                    recap_token_num, recap_after_negtive_num= min(recap_token_num + self.config['add_each_recap'],self.config['max_recap_token_num']), min(recap_after_negtive_num+self.config['add_each_neg'], self.config['max_negative_num'])
                else:
                    # print(f"[DEBUG] 正常推测思考流程")
                    if self.help_think_word_ids is not None:
                        generated_ids.extend(self.help_think_word_ids)
                    # 动态更新sampling参数，确保不超过总限制
                    current_len = len(generated_ids)
                    
                    # 检查当前长度是否接近最大限制
                    if current_len >= self.max_model_len - 100:
                        print(f"[WARNING] 当前长度 {current_len} 接近最大限制 {self.max_model_len}，停止推测思考")
                        break
                        
                    tgt_sampling_params_cache["max_new_tokens"] = self.calculate_available_tokens(current_len, self.config['max_target_tokens'])
                    # print(f"[DEBUG] 调用speculative_model进行推测")
                    spe_ids = get_model_result(self.speculative_model, generated_ids, tgt_sampling_params_cache)
                    
                    # 检查返回结果是否为空
                    if not spe_ids:
                        print(f"[WARNING] speculative_model推测返回空结果")
                        break
                        
                    spe_token = self.tokenizer.decode(spe_ids, skip_special_tokens=True)
                    # print(f"[DEBUG] speculative_model生成: {spe_token}")
                    spe_sent = sentiment_analysis(spe_token, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                    # print(f"[DEBUG] speculative情感分析: {spe_sent}")
                    if self.not_reasoning or spe_sent != 0:
                        # print(f"[DEBUG] 需要纠正，调用target_model")
                        try_correct_num = try_correct_num+1
                        # 动态更新sampling参数，确保不超过总限制
                        current_len = len(generated_ids)
                        
                        # 再次检查长度
                        if current_len >= self.max_model_len - 100:
                            print(f"[WARNING] 当前长度 {current_len} 接近最大限制 {self.max_model_len}，停止纠正")
                            break
                            
                        tgt_sampling_params_cache["max_new_tokens"] = self.calculate_available_tokens(current_len, self.config['max_target_tokens'])
                        tgt_ids = get_model_result(self.target_model, generated_ids, tgt_sampling_params_cache)
                        
                        # 检查返回结果是否为空
                        if not tgt_ids:
                            print(f"[WARNING] target_model纠正返回空结果")
                            break
                            
                        tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                        # print(f"[DEBUG] target_model生成: {tgt_token}")
                        tgt_sent = sentiment_analysis(tgt_token, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                        # print(f"[DEBUG] target情感分析: {tgt_sent}")
                        if self.choose_large or (spe_sent<0 and tgt_sent >=0) or (spe_sent>0 and tgt_sent<0):
                            # print(f"[DEBUG] 选择target_model结果")
                            decode_text = tgt_token
                            correct_tokens.append({
                                'pos': len(generated_ids)-prompt_len, 'token_num':self.config['max_target_tokens'],
                                'traget':tgt_token, 'speculative':spe_token})
                            generated_ids.extend(tgt_ids)
                            final_sent=tgt_sent
                        else:
                            # print(f"[DEBUG] 选择speculative_model结果")
                            generated_ids.extend(spe_ids)
                            decode_text = spe_token
                            final_sent=spe_sent
                        if final_sent < 0: negative_sent_num = negative_sent_num+1
                        if contains_keywords(decode_text, self.TARGET_VALIDATION_KEYWORDS['verify']):
                            change_tokens = self.config['original_recap_token_num']
                            change_flag = True
                if change_flag:
                    # print(f"[DEBUG] 执行change_flag逻辑")
                    try_correct_num = try_correct_num+1
                    # 动态计算可用的token数量，取change_tokens和剩余可用token的较小值
                    current_len = len(generated_ids)
                    
                    # 检查当前长度是否接近最大限制
                    if current_len >= self.max_model_len - 100:
                        print(f"[WARNING] 当前长度 {current_len} 接近最大限制 {self.max_model_len}，停止change操作")
                        break
                    
                    print(f"[DEBUG] current_len: {current_len}, max_model_len: {self.max_model_len}")
                    # 修复：使用模型最大长度而不是max_tokens参数来计算可用token数
                    available_tokens = self.calculate_available_tokens(current_len, self.max_model_len)
                    print(f"[DEBUG] available_tokens: {available_tokens}")
                    actual_change_tokens = min(change_tokens, available_tokens)
                    print(f"[DEBUG] actual_change_tokens: {actual_change_tokens}")
                    # print(f"[DEBUG] change_tokens={change_tokens}, available_tokens={available_tokens}, actual_change_tokens={actual_change_tokens}")
                    tgt_sampling_params_ = {
                        "max_new_tokens": actual_change_tokens, 
                        "temperature": temperature, 
                        "top_k": top_k, 
                        "top_p": top_p
                    } 
                    # print(f"[DEBUG] 调用target_model进行change")
                    tgt_ids = get_model_result(self.target_model, generated_ids, tgt_sampling_params_)
                    
                    # 检查返回结果是否为空
                    if not tgt_ids:
                        print(f"[WARNING] target_model change返回空结果")
                        break
                        
                    tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                    # print(f"[DEBUG] target_model change生成: {tgt_token}")
                    correct_tokens.append({
                        'pos': len(generated_ids)-prompt_len, 'token_num':change_tokens,
                        'traget':tgt_token})
                    generated_ids.extend(tgt_ids)
                    change_flag = False
            token_num = len(generated_ids)
            # print(f"[DEBUG] 当前token_num: {token_num}, max_tokens: {max_tokens}")
            if self.tokenizer.eos_token_id in generated_ids[-self.config['max_target_tokens']:]: 
                # print(f"[DEBUG] 遇到EOS token，跳出循环")
                break
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print(f"[DEBUG] 最终生成文本长度: {len(generated_text)}")
        # print(f"[DEBUG] 返回结果: tokens={len(generated_ids)-prompt_len}, correct_tokens={len(correct_tokens)}, try_correct_num={try_correct_num}")
        
        if return_only_generated:
            # 只返回生成的部分，不包括prompt
            generated_only_text = self.tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)
            return generated_only_text, len(generated_ids)-prompt_len, correct_tokens, try_correct_num
        
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

    model.generate(messages, 1024)
