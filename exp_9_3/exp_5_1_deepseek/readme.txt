1、填写flash_attention_triton_opt.py

2、完成 Flash Attention 算子的替换，修改 opt/tools/native/transformers_mlu/src/transformers/models/llama/modeling_llama.py文件。

3、运行DeepSeek聊天机器人程序，从而验证在 DLP 平台上替换 Flash Attention 算子后 DeepSeek实现的正确性。

    #运行脚本文件
    bash run_deepseek.sh

