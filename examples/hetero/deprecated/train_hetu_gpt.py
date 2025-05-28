from tqdm import tqdm
import os
import math
import logging
import hetu as ht
from hetu_gpt2 import GPTLMHeadModel
from gpt_config import GPTConfig
from load_data import DataLoaderForGPT
import numpy as np
import time
import argparse

def pretrain(args):
    # device_id=args.gpu_id
    # executor_ctx = ht.gpu(device_id)

    num_epochs = args.epochs
    lr = args.lr
    ht.init_comm_group(1, [7])
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    print(f"all_devices: {all_devices}")
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')

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
                    global_batch_size=args.train_batch_size,
                    )

    # Input data file names definition
    dict_seqlen2predlen = {128:20, 512:80}
    pred_len = dict_seqlen2predlen[config.max_position_embeddings]
    dataset = args.dataset
    if dataset not in ['wikicorpus_en', 'wiki_books']:
        raise(NotImplementedError)
    file_dir = './data/'    
    file_name_format = dataset + '_training_%d.hdf5'
    train_file_num = 1
    train_files = [file_dir + file_name_format%file_id for file_id in range(train_file_num)]

    # Hetu model definition
    model = GPTLMHeadModel(config=config)

    input_ids = ht.placeholder(ht.int64, shape=[config.global_batch_size, config.seq_len], device_groups=[all_devices])
    token_type_ids = ht.placeholder(ht.int64, shape=[config.global_batch_size, config.seq_len], device_groups=[all_devices]) 
    attention_mask = ht.placeholder(ht.float32, shape=[config.global_batch_size, config.seq_len], device_groups=[all_devices]) 

    masked_lm_labels = ht.placeholder(ht.int64, shape=[config.global_batch_size, config.seq_len], device_groups=[all_devices]) 
    next_sentence_label = ht.placeholder(ht.int64, shape=[config.global_batch_size, config.seq_len], device_groups=[all_devices]) 

    loss_position_sum = ht.placeholder(ht.float32, shape=[config.global_batch_size, config.seq_len], device_groups=[all_devices]) 

    loss, lm_logits = model(input_ids=input_ids, 
                            token_type_ids=token_type_ids, 
                            attention_mask=attention_mask, 
                            labels=masked_lm_labels)
    
    loss_mean = ht.div(loss, loss_position_sum)

    # masked_lm_loss_mean = ht.div(masked_lm_loss, loss_position_sum)
    # # next_sentence_loss_mean = ht.mean(next_sentence_loss, [0])
    # next_sentence_loss_mean = next_sentence_loss


    # loss = masked_lm_loss_mean + next_sentence_loss_mean
    # opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = args.adam_weight_decay)
    #opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    train_op = opt.minimize(loss_mean)
    # executor = ht.Executor([masked_lm_loss_mean, next_sentence_loss_mean, loss, train_op],ctx=executor_ctx,dynamic_memory=True)

    global_step_num = 0
    for ep in range(num_epochs):
        step_num = 0
        print("Epoch:", ep)
        for train_file in train_files:
            dataloader = DataLoaderForGPT(train_file, config.global_batch_size, pred_len)
            for i in range(dataloader.batch_num):
                start_time = time.time()
                batch_data = dataloader.get_batch(i)
                # print(batch_data['input_ids'].shape, batch_data['token_type_ids'].shape,
                #       batch_data['attention_mask'].shape, batch_data['masked_lm_labels'].shape,
                #       batch_data['next_sentence_label'].shape,
                #       np.array([np.where(batch_data['masked_lm_labels'].reshape(-1)!=-1)[0].shape[0]]).shape)
                # print(batch_data['input_ids'].dtype, batch_data['token_type_ids'].dtype,
                #       batch_data['attention_mask'].dtype, batch_data['masked_lm_labels'].dtype,
                #       batch_data['next_sentence_label'].dtype,
                #       np.array([np.where(batch_data['masked_lm_labels'].reshape(-1)!=-1)[0].shape[0]]).dtype)
                feed_dict = {
                    input_ids: batch_data['input_ids'].astype(np.int64).reshape([config.global_batch_size, 128]),
                    token_type_ids: batch_data['token_type_ids'].astype(np.int64).reshape([config.global_batch_size, 128]),
                    attention_mask: batch_data['attention_mask'].astype(np.float32).reshape([config.global_batch_size, 128]),
                    masked_lm_labels: batch_data['masked_lm_labels'].astype(np.int64).reshape([config.global_batch_size, 128]),
                    next_sentence_label: batch_data['next_sentence_label'].astype(np.int64).reshape([config.global_batch_size, 1]),
                    loss_position_sum: np.array([np.where(batch_data['masked_lm_labels'].reshape(-1)!=-1)[0].shape[0]]).astype(np.float32),
                }
                results = train_op.graph.run(loss_mean, [loss_mean, lm_logits, train_op], feed_dict = feed_dict)
                loss_out = results[0].numpy(force=True)
                # print("LABEL:", results[1].numpy(force=True).sum(), " ", loss_out)
                # print(results[2].numpy(force=True), results[2].numpy(force=True).shape, loss_out)
                # assert(1==0)
                # next_sentence_loss_mean_out = results[1].numpy(force=True)
                # loss_out = results[2].numpy(force=True)
                end_time = time.time()
                print('[Epoch %d] (Iteration %d): Loss = %.3f, Time = %.3f'%(ep, step_num, loss_out, end_time-start_time))
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
    with ht.graph("define_and_run"):
        pretrain(args)