from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_mlu
import torch_mlu.utils.gpu_migration
import time
# import os
# # 检查环境变量 OPT_FA 是否为 "1"
# OPT_FA = os.environ.get('OPT_FA') == '1'
model_name = "/workspace/model/favorite/large-scale-models/model-v1/DeepSeek-R1-Distill-Llama-8B/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"
#TODO：使用Transformer库提供的接口从指定路径加载预训练语言模型，采用半精度浮点数格式加载，并自动分配到可用设备以提升推理效率
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")
#TODO：加载与模型对应的分词器，用于将文本编码为模型可识别的tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "我只有3000元，帮我制定一份北京3天游"
#TODO：构造符合聊天格式的消息列表，角色为user，内容是prompt文本
messages = [{"role": "user", "content": prompt}]
#TODO：将对话消息列表格式化为模型所需的对话模板文本，并添加生成提示符但不进行分词
text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
#TODO：将对话文本编码为PyTorch张量并移动到模型所在的设备上
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#TODO：记录推理开始时间
start_time = time.time()
#TODO：在该代码块内禁用梯度计算，以节省显存和加速推理（只用于推理，无需反向传播）
with torch.no_grad():
    #TODO：调用语言模型基于输入生成最多512个token的回复
    generated_ids = model.generate(**model_inputs,max_new_tokens=512)
#TODO：记录推理结束时间
end_time = time.time()
#TODO：通过截断去除输入部分，只保留模型生成的新token，得到最终用于解码的token序列
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
#TODO：将生成的token序列解码成文本，跳过特殊符号，并返回首个解码结果作为模型回复
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#TODO：输出模型生成的回复文本
print(response)
print("DeepSeek Time cost:", end_time - start_time)
# if OPT_FA == 0:
#     print("DeepSeek Time cost:", end_time - start_time)
# elif OPT_FA == 1:
#     print("DeepSeek OPT Time cost:", end_time - start_time)

