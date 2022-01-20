from tqdm import tqdm
import os
import math
import logging
import hetu as ht
from hetu_bert_ccfbdci_2128 import BertForPreTraining_mp, BertConfig
from load_data import DataLoader_example_data_for_bert
import numpy as np
import time
import torch
from tqdm import tqdm

'''
heturun -c config2.yml python train_hetu_bert_ccfbdci_2128.py
'''

num_epochs = 1
lr = 1e-4

hidden_size = 2128

config = BertConfig(vocab_size=30522, 
                    hidden_size=hidden_size,
                    num_hidden_layers=24, 
                    num_attention_heads=16, 
                    intermediate_size=hidden_size * 4, 
                    max_position_embeddings=512, 
                    batch_size=2)

model = BertForPreTraining_mp(config=config)

batch_size = config.batch_size
seq_len = config.max_position_embeddings
vocab_size = config.vocab_size

dataloader = DataLoader_example_data_for_bert(batch_size = batch_size)
data_names = ['input_ids','token_type_ids','attention_mask','masked_lm_labels','next_sentence_label']

with ht.context(ht.gpu(0)):
    input_ids = ht.Variable(name='input_ids', trainable=False)
    token_type_ids = ht.Variable(name='token_type_ids', trainable=False)
    attention_mask = ht.Variable(name='attention_mask', trainable=False)

    masked_lm_labels = ht.Variable(name='masked_lm_labels', trainable=False)
    next_sentence_label = ht.Variable(name='next_sentence_label', trainable=False)

    loss_position_sum = ht.Variable(name='loss_position_sum', trainable=False)

_,_, masked_lm_loss, next_sentence_loss = model(input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)

with ht.context(ht.gpu(0)):
    masked_lm_loss_mean = ht.div_op(ht.reduce_sum_op(masked_lm_loss, [0,1]), loss_position_sum)
    next_sentence_loss_mean = ht.reduce_mean_op(next_sentence_loss, [0])

with ht.context(ht.gpu(1)):
    loss = masked_lm_loss_mean + next_sentence_loss_mean
    opt = ht.optim.AdamWOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay = 0.01)
    # train_op = opt.minimize(loss)
    # executor = ht.Executor([masked_lm_loss_mean, next_sentence_loss_mean, loss, train_op],dynamic_memory=True)
    train_ops = opt.minimize_per_grad(loss)
    executor = ht.Executor([masked_lm_loss_mean, next_sentence_loss_mean, loss] + train_ops,dynamic_memory=True)

dataloader.make_epoch_data()
for ep in range(num_epochs):
    for i in tqdm(range(dataloader.batch_num)):
        if executor.rank == 0:
            batch_data = dataloader.get_batch(i)
            feed_dict = {
                input_ids: batch_data['input_ids'],
                token_type_ids: batch_data['token_type_ids'],
                attention_mask: batch_data['attention_mask'],
                masked_lm_labels: batch_data['masked_lm_labels'],
                next_sentence_label: batch_data['next_sentence_label'],
                loss_position_sum: np.array([np.where(batch_data['masked_lm_labels'].reshape(-1)!=-1)[0].shape[0]]),
            }
        
            results = executor.run(feed_dict = feed_dict)
        else:
            results = executor.run()