from transformers import AutoModel, AutoTokenizer

# 加载模型和分词器
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 模型路径或名称，假设它在Hugging Face上
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 示例文本
text = "你好，世界！"

# 分词
inputs = tokenizer(text, return_tensors="pt")

# 获取模型输出
outputs = model(**inputs)

# 查看输出
print(outputs)
