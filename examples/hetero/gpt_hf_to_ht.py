import torch
from collections import OrderedDict


state_dict = torch.load("./checkpoint/HuggingFace/pytorch_model.bin")
# state_dict = torch.load("./checkpoint/Megatron/pytorch_model.bin")
new_state_dict = OrderedDict()


hf_to_ht = {
    "attn.c_attn": "attn.qkv_dense",
    "attn.c_proj": "attn.dense",
    "mlp.c_fc": "mlp.parallel_mlp.dense_h_to_4h",
    "mlp.c_proj": "mlp.parallel_mlp.dense_4h_to_h",
    "wte.weight": "wte.embedding_table",
    "wpe.weight": "wpe.embedding_table",
}


# TODO: make it a argument that determined by the TP size
vocab_padding_size = 1


# Megatron padding approach: replicate the final entry
def padding_vocab_embedding(key, value):
    print(f"Pad the {key} shape", [value.shape[0], value.shape[1]], 
          "to", [value.shape[0] + vocab_padding_size, value.shape[1]])
    return torch.cat((
            value,
            value[-1].unsqueeze(0).expand(vocab_padding_size, -1)))


for key in state_dict:
    if "wte.weight" in key or "lm_head.weight" in key:
        state_dict[key] = padding_vocab_embedding(key, state_dict[key])
        
    value = state_dict[key]
    
    for subkey in hf_to_ht:
        key = key.replace(subkey, hf_to_ht[subkey])
    if key != "lm_head.weight" and "transformer." not in key:
        key = "transformer." + key
    if "attn.bias" in key:
        continue
    new_state_dict[key] = value
    if "dense" in key and "weight" in key:
        new_state_dict[key] = new_state_dict[key].T
        
if "lm_head.weight" not in new_state_dict.keys():
    new_state_dict["lm_head.weight"] = state_dict["wte.weight"]


print(new_state_dict.keys())
torch.save(new_state_dict, "./checkpoint/HuggingFace/hetu_pytorch_model.bin")
# torch.save(new_state_dict, "./checkpoint/Megatron/hetu_pytorch_model.bin")

    
    