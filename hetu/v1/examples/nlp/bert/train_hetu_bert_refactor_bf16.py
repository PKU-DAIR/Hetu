from tqdm import tqdm
import os
import math
import logging
import hetu as ht
from hetu_bert_refactor import BertForPreTraining
from bert_config import BertConfig
from load_data import DataLoaderForBertPretraining
import numpy as np
import time
import argparse

def pretrain(args):
    # device_id=args.gpu_id
    # executor_ctx = ht.gpu(device_id)

    num_epochs = args.epochs
    lr = args.lr

    config = BertConfig(vocab_size=args.vocab_size, 
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        max_position_embeddings=args.seq_length, 
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob,
                        batch_size=args.train_batch_size,
                        hidden_act=args.hidden_act)

    # Input data file names definition
    dict_seqlen2predlen = {128:20, 512:80}
    pred_len = dict_seqlen2predlen[config.max_position_embeddings]
    dataset = args.dataset
    if dataset not in ['wikicorpus_en', 'wiki_books']:
        raise(NotImplementedError)
    file_dir = './data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/%s/'%dataset
    file_name_format = dataset + '_training_%d.hdf5'
    train_file_num = 16
    train_files = [file_dir + file_name_format%file_id for file_id in range(train_file_num)]

    # Hetu model definition
    model = BertForPreTraining(config=config)
    model1 = model.clone(dtype=ht.bfloat16)

    input_ids = ht.placeholder(ht.int64, shape=[config.batch_size, 128]) #ht.Variable(name='input_ids', trainable=False)
    token_type_ids = ht.placeholder(ht.int64, shape=[config.batch_size, 128]) #ht.Variable(name='token_type_ids', trainable=False)
    attention_mask = ht.placeholder(ht.bfloat16, shape=[config.batch_size, 128]) #ht.Variable(name='attention_mask', trainable=False)
    attention_mask32 = ht.placeholder(ht.float32, shape=[config.batch_size, 128]) #ht.Variable(name='attention_mask', trainable=False)
    
    masked_lm_labels = ht.placeholder(ht.int64, shape=[config.batch_size, 128]) #ht.Variable(name='masked_lm_labels', trainable=False)
    next_sentence_label = ht.placeholder(ht.int64, shape=[config.batch_size]) #ht.Variable(name='next_sentence_label', trainable=False)

    loss_position_sum32 = ht.placeholder(ht.float32, shape=[config.batch_size]) #ht.Variable(name='loss_position_sum', trainable=False)
    loss_position_sum = ht.placeholder(ht.bfloat16, shape=[config.batch_size]) #ht.Variable(name='loss_position_sum', trainable=False)

    _, _, masked_lm_loss, next_sentence_loss = model1(input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)
    
    masked_lm_loss_mean =  ht.div(masked_lm_loss, loss_position_sum)
    next_sentence_loss_mean = ht.mean(next_sentence_loss, [0])
    next_sentence_loss_mean = next_sentence_loss

    loss = masked_lm_loss_mean + next_sentence_loss_mean

    opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    train_op = opt.minimize(loss)

    _, _, masked_lm_loss32, next_sentence_loss32 = model(input_ids, token_type_ids, attention_mask32, masked_lm_labels, next_sentence_label)
 
    masked_lm_loss_mean32 =  ht.div(masked_lm_loss32, loss_position_sum32)
    next_sentence_loss_mean32 = ht.mean(next_sentence_loss32, [0])
    next_sentence_loss_mean32 = next_sentence_loss32
    
    
    loss32 = masked_lm_loss_mean32 + next_sentence_loss_mean32
    opt2 = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    train_op2 = opt2.minimize(loss32)
    # # opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = args.adam_weight_decay)
    # #opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

    # executor = ht.DARExecutor("cuda:0")
    # executor = ht.Executor([masked_lm_loss_mean, next_sentence_loss_mean, loss, train_op],ctx=executor_ctx,dynamic_memory=True)

    global_step_num = 0
    for ep in range(num_epochs):
        step_num = 0
        for train_file in train_files:
            dataloader = DataLoaderForBertPretraining(train_file, config.batch_size, pred_len)
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
                    input_ids: batch_data['input_ids'].astype(np.int64).reshape([config.batch_size, 128]),
                    token_type_ids: batch_data['token_type_ids'].astype(np.int64).reshape([config.batch_size, 128]),
                    attention_mask32: batch_data['attention_mask'].astype(np.float32).reshape([config.batch_size, 128]),
                    attention_mask: batch_data['attention_mask'].astype(np.float32).reshape([config.batch_size, 128]),
                    masked_lm_labels: batch_data['masked_lm_labels'].astype(np.int64).reshape([config.batch_size, 128]),
                    next_sentence_label: batch_data['next_sentence_label'].astype(np.int64).reshape([config.batch_size,]),
                    loss_position_sum32: np.array([np.where(batch_data['masked_lm_labels'].reshape(-1)!=-1)[0].shape[0]]).astype(np.float32),
                    loss_position_sum: np.array([np.where(batch_data['masked_lm_labels'].reshape(-1)!=-1)[0].shape[0]]).astype(np.float32),
                }
                results = train_op.graph.run([masked_lm_loss_mean, next_sentence_loss_mean, loss, train_op], feed_dict = feed_dict)
                # results = executor.run([masked_lm_loss_mean, next_sentence_loss_mean, loss, train_op], feed_dict = feed_dict)
                masked_lm_loss_mean_out = results[0].numpy(force=True)
                next_sentence_loss_mean_out = results[1].numpy(force=True)
                loss_out = results[2].numpy(force=True)
                end_time = time.time()
                print('[Epoch %d] (Iteration %d): Loss = %.3f, MLM_loss = %.3f, NSP_loss = %.6f, Time = %.3f'%(ep,step_num,loss_out, masked_lm_loss_mean_out, next_sentence_loss_mean_out, end_time-start_time))
                # results32 = train_op.graph.run([masked_lm_loss_mean32, next_sentence_loss_mean32, loss32], feed_dict = feed_dict)
                # # results = executor.run([masked_lm_loss_mean, next_sentence_loss_mean, loss, train_op], feed_dict = feed_dict)
                # masked_lm_loss_mean_out32 = results32[0].numpy(force=True)
                # next_sentence_loss_mean_out32 = results32[1].numpy(force=True)
                # loss_out32 = results32[2].numpy(force=True)
                # end_time = time.time()
                # print('[Epoch %d] (Iteration %d): Loss = %.3f, MLM_loss = %.3f, NSP_loss = %.6f, Time = %.3f'%(ep,step_num,loss_out32, masked_lm_loss_mean_out32, next_sentence_loss_mean_out32, end_time-start_time))
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