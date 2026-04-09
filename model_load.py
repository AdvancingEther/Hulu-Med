import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

hulu_4b_path = "/home/deeplearning/data/data2/wzc/VolInterp/methods/checkpoints/Hulu-Med-4B"
hulu_7b_path = "/home/deeplearning/data/data2/wzc/VolInterp/methods/checkpoints/Hulu-Med-7B"

model_path = hulu_7b_path

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
).eval()

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

tokenizer = processor.tokenizer
print(type(model), type(processor), type(tokenizer))