import torch
from collections import OrderedDict


state_dict = torch.load("./checkpoint/HuggingFace/pytorch_model.bin")
# state_dict = torch.load("./checkpoint/Megatron/pytorch_model.bin")

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

torch.save(state_dict, "./checkpoint/HuggingFace_new/pytorch_model.bin")
# torch.save(new_state_dict, "./checkpoint/Megatron_new/pytorch_model.bin")

    
    