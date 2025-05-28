import torch
import ptvsd # used for debug

# ptvsd.enable_attach(address =('127.0.0.1', 4000))
# ptvsd.wait_for_attach()

origin = torch.load('checkpoint/HuggingFace/hetu_pytorch_model.bin')
x_1 = torch.load('checkpoint/temp/hetu_pytorch_model-1-of-2.bin')
x_2 = torch.load('checkpoint/temp/hetu_pytorch_model-2-of-2.bin')
y = torch.load('checkpoint/temp/hf_pytorch_model.bin')

assert(abs(y['transformer.h.0.ln_2.weight'].sum() - x_1['transformer.h.0.ln_2.weight'].sum()) < 0.1)
assert(abs(y['transformer.h.10.attn.c_attn.weight'].sum() - x_2['transformer.h.10.attn.qkv_dense.weight'].sum()) < 0.1)

print('precision correct!')