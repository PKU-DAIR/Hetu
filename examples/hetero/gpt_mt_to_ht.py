import torch
from collections import OrderedDict
import ptvsd

ptvsd.enable_attach(address =('127.0.0.1', 4000))
ptvsd.wait_for_attach()

# state_dict = torch.load("./checkpoint/Megatron/model_optim_rng.pt")
state_dict = torch.load("./checkpoint/Megatron/pytorch_model.bin")

# TODO: implementation

print("Please use gpt_mt_to_hf.py, and then use gpt_hf_to_ht.py for now!")

