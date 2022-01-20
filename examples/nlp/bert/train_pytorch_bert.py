from tqdm import tqdm
import os
import math
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import numpy as np
from pytorch_bert import BertModel, BertForPreTraining
from bert_config import BertConfig
from load_data import DataLoader
import time

''' Usage example:
    In dir Hetu/examples/nlp/bert/: python train_pytorch_bert.py
'''

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

num_epochs = 1
lr = 1e-4

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda_condition else "cpu")

# config = BertConfig(vocab_size=30522, 
#                     hidden_size=768,
#                     num_hidden_layers=12, 
#                     num_attention_heads=12, 
#                     intermediate_size=3072, 
#                     max_position_embeddings=512, 
#                     attention_probs_dropout_prob=0.0,
#                     hidden_dropout_prob=0.0,
#                     batch_size=18)

hidden_size = 1600

config = BertConfig(vocab_size=30522, 
                    hidden_size=hidden_size,
                    num_hidden_layers=24, 
                    num_attention_heads=16, 
                    intermediate_size=hidden_size * 4, 
                    max_position_embeddings=512, 
                    # attention_probs_dropout_prob=0.0,
                    # hidden_dropout_prob=0.0,
                    batch_size=18)

model = BertForPreTraining(config=config)
model.to(device)

batch_size = config.batch_size
seq_len = config.max_position_embeddings
vocab_size = config.vocab_size

dataloader = DataLoader(dataset='bookcorpus', doc_num=200, save_gap=200, batch_size = batch_size)
data_names = ['input_ids','token_type_ids','attention_mask','masked_lm_labels','next_sentence_label']

#init parameters
for m in model.modules():
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_normal_(m.weight)

# # save init parameters
# params = model.state_dict()
# for key, val in params.items():
#     params[key] = val.cpu().numpy()
# torch.save(params, "pytorch_params.file") 

opt = Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8, weight_decay = 0.01)
# opt = Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8)
# opt = SGD(model.parameters(), lr=lr)

# # load parameters
# load_ep = 0.0
# load_i = 5
# load_path = './pretrained_params/pytorch_pretrained_params/'
# load_file = 'epoch_%d_iter_%d.params'%(load_ep,load_i)
# state_dict = torch.load(load_path+load_file, map_location='cpu' if not torch.cuda.is_available() else None)
# model.load_state_dict(state_dict)

dataloader.make_epoch_data()
for ep in range(num_epochs):
    for i in range(dataloader.batch_num):
        start_time = time.time()

        batch_data = dataloader.get_batch(i)
        input_ids = torch.LongTensor(batch_data['input_ids']).to(device)
        token_type_ids = torch.LongTensor(batch_data['token_type_ids']).to(device)
        attention_mask = torch.LongTensor(batch_data['attention_mask']).to(device)
        masked_lm_labels = torch.LongTensor(batch_data['masked_lm_labels']).to(device)
        next_sentence_label = torch.LongTensor(batch_data['next_sentence_label']).to(device)
        
        opt.zero_grad()
        _,_, masked_lm_loss_mean, next_sentence_loss_mean = model(input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)   
        loss = masked_lm_loss_mean + next_sentence_loss_mean        
        loss.backward()
        opt.step()

        # # save parameters
        # if i%5000 == 0 and i != 0:
        #     save_path = './pretrained_params/pytorch_pretrained_params_adam/'
        #     save_file = 'epoch_%d_iter_%d.params'%(ep,i)
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     torch.save(model.state_dict(), save_path+save_file) 

        masked_lm_loss_out = masked_lm_loss_mean.item()
        next_sentence_loss_out = next_sentence_loss_mean.item()
        loss_out = loss.item()

        end_time = time.time()
        print('[Epoch %d] (Iteration %d): Loss = %.3f, MLM_loss = %.3f, NSP_loss = %.6f, Time = %.3f'%(ep,i,loss_out, masked_lm_loss_out, next_sentence_loss_out, end_time-start_time))
