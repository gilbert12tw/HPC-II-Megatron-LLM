#!/bin/python3
from transformers import AutoTokenizer, AutoConfig

model_name = "meta-llama/Llama-2-7b-hf" # 或其他 Llama 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# 保存 tokenizer 文件 (通常是 tokenizer.model)
tokenizer.save_pretrained('./llama2_tokenizer')
# config 文件也可用於參考模型架構參數
config.save_pretrained('./llama2_config')

# 預處理數據時 --vocab-file 就指向 ./llama2_tokenizer/tokenizer.model
