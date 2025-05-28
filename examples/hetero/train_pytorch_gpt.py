from tqdm import tqdm
import os
import math
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import numpy as np
from pytorch_gpt import GPTModel, GPTLMHeadModel
from gpt_config import GPTConfig
from load_data import DataLoaderForGPT
import time
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(12345)

def pretrain(args):
    num_epochs = args.epochs
    lr = args.lr

    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")

    config = GPTConfig(vocab_size=args.vocab_size, 
                       n_positions=args.seq_length,
                       n_ctx=args.seq_length,
                       n_embd=args.hidden_size,
                       n_layer=args.num_hidden_layers, 
                       n_head=args.num_attention_heads, 
                       # n_inner=4*args.hidden_size,
                       resid_pdrop=args.dropout_prob,
                       embd_pdrop=args.dropout_prob,
                       attn_pdrop=args.dropout_prob,
                       activation_function=args.hidden_act,
                       batch_size=args.train_batch_size,
                      )

    model = GPTLMHeadModel(config=config)
    model.to(device)

    # Input data file names definition
    dict_seqlen2predlen = {128:20, 512:80}
    pred_len = dict_seqlen2predlen[config.max_position_embeddings]
    dataset = args.dataset
    if dataset not in ['wikicorpus_en', 'wiki_books']:
        raise(NotImplementedError)
    file_dir = '../bert/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/%s/'%dataset
    file_name_format = dataset + '_training_%d.hdf5'
    train_file_num = 256
    train_files = [file_dir + file_name_format%file_id for file_id in range(train_file_num)]
    print(train_files)

    #init parameters
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(m.weight)

    # # save init parameters
    # params = model.state_dict()
    # for key, val in params.items():
    #     params[key] = val.cpu().numpy()
    # torch.save(params, "pytorch_params.file") 

    # opt = Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8, weight_decay = args.adam_weight_decay)
    # opt = Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8)
    opt = SGD(model.parameters(), lr=lr)

    # # load parameters
    # load_ep = 0.0
    # load_i = 5
    # load_path = './pretrained_params/pytorch_pretrained_params/'
    # load_file = 'epoch_%d_iter_%d.params'%(load_ep,load_i)
    # state_dict = torch.load(load_path+load_file, map_location='cpu' if not torch.cuda.is_available() else None)
    # model.load_state_dict(state_dict)

    global_step_num = 0
    for ep in range(num_epochs):
        step_num = 0
        for train_file in train_files:
            dataloader = DataLoaderForGPT(train_file, config.batch_size, pred_len)
            for i in range(dataloader.batch_num):
                start_time = time.time()
                batch_data = dataloader.get_batch(i)
                input_ids = torch.LongTensor(batch_data['input_ids']).to(device)
                token_type_ids = torch.LongTensor(batch_data['token_type_ids']).to(device)
                attention_mask = torch.LongTensor(batch_data['attention_mask']).to(device)
                masked_lm_labels = torch.LongTensor(batch_data['masked_lm_labels']).to(device)
                next_sentence_label = torch.LongTensor(batch_data['next_sentence_label']).to(device)
                opt.zero_grad()
                loss, lm_logits, transformer_output= model(input_ids=input_ids, 
                                                           token_type_ids=token_type_ids, 
                                                           attention_mask=attention_mask, 
                                                           labels=masked_lm_labels)   
                # loss = masked_lm_loss_mean + next_sentence_loss_mean   
                loss.backward()
                opt.step()

                # # save parameters
                # if i%5000 == 0 and i != 0:
                #     save_path = './pretrained_params/pytorch_pretrained_params_adam/'
                #     save_file = 'epoch_%d_iter_%d.params'%(ep,i)
                #     if not os.path.exists(save_path):
                #         os.makedirs(save_path)
                #     torch.save(model.state_dict(), save_path+save_file) 

                # masked_lm_loss_out = masked_lm_loss_mean.item()
                # next_sentence_loss_out = next_sentence_loss_mean.item()
                loss_out = loss.item()
                # transformer_out = transformer_output[0].mean().item()


                end_time = time.time()
                print('[Epoch %d] (Iteration %d/%d): Loss = %.3f, Time = %.3f'%(ep,step_num,dataloader.batch_num,loss_out, end_time-start_time))
                step_num += 1
                global_step_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--dataset", type=str, default='wikicorpus_en', help="Dataset used to train."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="Hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    args = parser.parse_args()

    pretrain(args)