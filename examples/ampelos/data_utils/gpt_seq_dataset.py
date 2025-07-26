import os
import json
import pickle
import time
from tqdm import tqdm
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .tokenizer import build_tokenizer, ChatGLM4Tokenizer
from transformers.tokenization_utils import AddedToken
from transformers.utils import PaddingStrategy
from PIL import Image
from collections import deque
import threading
import multiprocessing


class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = build_tokenizer(self.args)

    def pad_id(self):
        return self.tokenizer.pad

    def encode(self, json_line):
        data = json.loads(json_line)
        doc = data[self.args.key] # key: content for web, text for wiki
        assert (self.args.tokenizer_type == 'GPT2BPETokenizer'), 'Now only support GPT2BPETokenizer!'
        doc_ids = self.tokenizer.tokenize(doc)
        
        return doc_ids

class GPTJsonDataset(Dataset):
    def __init__(self, json_file, key, max_seq_len, vocab_file, merge_file):
        args = {
            'key': key,
            'rank': 0,
            'make_vocab_size_divisible_by': 128,
            'tensor_model_parallel_size': 1,
            'vocab_extra_ids': 0,
            'tokenizer_type': 'GPT2BPETokenizer',
            'vocab_file': vocab_file,
            'merge_file': merge_file,
        }
        args = SimpleNamespace(**args)
        self.encoder = Encoder(args)
        self.data = []

        cache_path = json_file.split('.')[0] + f'_cache.pkl'
        if os.path.exists(cache_path) and 1 == 0:
            # read exists data cache here
            print(f'Loading exists cache from {cache_path} begin ...')
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            end_time = time.time()
            print(f'Loading exists cache end, time cost: {end_time - start_time: .3f} s')
        else:
            # tokenize data from json file
            print(f'Building dataset begin ...')
            start_time = time.time()
            with open(json_file, 'r') as f:
                for json_line in tqdm(f.readlines()):
                    doc_ids = self.encoder.encode(json_line)
                    # doc_ids may be empty, will cause error when mbs=1
                    if len(doc_ids) > 0:
                        self.data.append(doc_ids)
            # save cache
            # with open(cache_path, 'wb') as f:
            #     pickle.dump(self.data, f)                    
            end_time = time.time()
            print(f'Building dataset end, time cost: {end_time - start_time: .3f} s')

        # deal with max_seq_len + 1 (for tokens/labels seq_len = max_seq_len+1 - 1)
        print(f'Cutting or padding data to max_seq_len + 1 = {max_seq_len + 1} begin ...')
        start_time = time.time()
        max_seq_len = max_seq_len + 1
        for idx, doc_ids in enumerate(self.data):
            if len(doc_ids) > max_seq_len:
                self.data[idx] = doc_ids[:max_seq_len]
            elif len(doc_ids) < max_seq_len:
                self.data[idx] += [self.encoder.pad_id()] * (max_seq_len - len(doc_ids))
        end_time = time.time()
        print(f'Cutting or padding data end, time cost: {end_time - start_time: .3f} s')

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tokens = torch.tensor(self.data[idx])
        tokens = np.array(self.data[idx])
        return tokens

class LLaMAJsonDataset(Dataset):
    def __init__(self, json_file, key, max_seq_len, vocab_file, merge_file, eod_vocab = '<|endoftext|>'):
        args = {
            'key': key,
            'rank': 0,
            'make_vocab_size_divisible_by': 128,
            'tensor_model_parallel_size': 1,
            'vocab_extra_ids': 0,
            'tokenizer_type': 'GPT2BPETokenizer',
            'vocab_file': vocab_file,
            'merge_file': merge_file,
            'eod_vocab': eod_vocab,
        }
        args = SimpleNamespace(**args)
        self.data = []

        cache_path = json_file.split('.')[0] + f'_llama_cache_seqlen_' + str(max_seq_len)  + '.pkl'
        from transformers import LlamaTokenizer
        with open('data_utils/tokenizer/models/llama2/tokenizer_config.json', 'r', encoding='utf-8') as file:  
            tokenizer_config = json.load(file)
        for k, v in tokenizer_config.items():  
            if isinstance(v, dict) and "__type" in v.keys() and v["__type"] == "AddedToken":
                v.pop("__type")
                tokenizer_config[k] = AddedToken(**v)
            
        tokenizer = LlamaTokenizer(vocab_file="data_utils/tokenizer/models/llama2/tokenizer.model",
                                   **tokenizer_config)
        self.encoder = tokenizer
        if os.path.exists(cache_path):
            # read exists data cache here
            print(f'Loading exists cache from {cache_path} begin ...')
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            end_time = time.time()
            print(f'Loading exists cache end, time cost: {end_time - start_time: .3f} s')
        else:
            # tokenize data from json file
            print(f'Building dataset begin ...')
            start_time = time.time()
            with open(json_file, 'r') as f:
                for json_line in tqdm(f.readlines()):
                    print("JSON:", json_line[key])
                    exit(0)
                    doc_ids = self.encoder.encode(json_line[key])
                    # doc_ids may be empty, will cause error when mbs=1
                    if len(doc_ids) > 0:
                        self.data.append(doc_ids)
                        
            # save cache
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data, f)                    
            end_time = time.time()
            print(f'Building dataset end, time cost: {end_time - start_time: .3f} s')

        # deal with max_seq_len + 1 (for tokens/labels seq_len = max_seq_len+1 - 1)
        print(f'Cutting or padding data to max_seq_len + 1 = {max_seq_len + 1} begin ...')
        start_time = time.time()
        max_seq_len = max_seq_len + 1
        for idx, doc_ids in enumerate(self.data):
            if len(doc_ids) > max_seq_len:
                self.data[idx] = doc_ids[:max_seq_len]
            elif len(doc_ids) < max_seq_len:
                self.data[idx] += [self.encoder.eos_token_id] * (max_seq_len - len(doc_ids))
        end_time = time.time()
        print(f'Cutting or padding data end, time cost: {end_time - start_time: .3f} s')

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tokens = torch.tensor(self.data[idx])
        tokens = np.array(self.data[idx])
        return tokens

class GLMJsonDataset(Dataset):
    def __init__(self, json_file, key, max_seq_len, 
                 image_size, num_channels, vocab_file, merge_file):
        args = {
            'key': key,
            'rank': 0,
            'make_vocab_size_divisible_by': 128,
            'tensor_model_parallel_size': 1,
            'vocab_extra_ids': 0,
            'tokenizer_type': 'GPT2BPETokenizer',
            'vocab_file': vocab_file,
            'merge_file': merge_file,
        }
        args = SimpleNamespace(**args)
        self.encoder = Encoder(args)
        with open('data_utils/tokenizer/models/glm/tokenizer_config.json', 'r', encoding='utf-8') as file:  
            tokenizer_config = json.load(file)
        # print("TKC:", type(tokenizer_config))
        string_keys = []
        for k, v in tokenizer_config["added_tokens_decoder"].items():  
            if isinstance(v, dict):
                tokenizer_config["added_tokens_decoder"][k] = AddedToken(**v)
                string_keys.append(k)
        for k in string_keys:
            tokenizer_config["added_tokens_decoder"][int(k)] = tokenizer_config["added_tokens_decoder"][k]
            tokenizer_config["added_tokens_decoder"].pop(k)
        
        tokenizer_config["image_size"] = image_size
            
        self.encoder = ChatGLM4Tokenizer(vocab_file="data_utils/tokenizer/models/glm/tokenizer.model",
                                         **tokenizer_config)
        self.data = []
        self.image_size = image_size
        self.num_channels = num_channels

        cache_path = json_file.split('.')[0] + f'_glm_cache.pkl'
        if os.path.exists(cache_path):
            # read exists data cache here
            print(f'Loading exists cache from {cache_path} begin ...')
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            end_time = time.time()
            print(f'Loading exists cache end, time cost: {end_time - start_time: .3f} s')
        else:
            # tokenize data from json file
            print(f'Building dataset begin ...')
            start_time = time.time()
            ptr = 0
            with open(json_file, 'r') as f:
                for json_line in tqdm(f.readlines()):
                    input_txt = json.loads(json_line)[args.key]
                    ptr += 1
                    if (ptr > 100):
                        break
                    images = np.random.randint(0, 255, (image_size, image_size, num_channels)).astype(np.uint8)
                    images = Image.fromarray(images)
                    doc_ids = self.encoder.apply_chat_template([{"role": "user", "image": images, "content": input_txt}],
                                                                max_length=max_seq_len + 1, 
                                                                padding=True, truncation=True,
                                                                add_generation_prompt=True, tokenize=True, 
                                                                return_tensors="pt", return_dict=True)
                    
                    # doc_ids = self.encoder.encode(json_line)
                    doc_ids = self.encoder.pad(doc_ids, padding=PaddingStrategy.MAX_LENGTH, 
                                               max_length=max_seq_len + 1)
                    # doc_ids may be empty, will cause error when mbs=1
                    input_id = doc_ids['input_ids'].tolist()[0]
                    boi_token_pos, eoi_token_pos = input_id.index(151339), input_id.index(151340)
                    begin_end = [boi_token_pos, eoi_token_pos]
                    begin_end = torch.from_numpy(np.array(begin_end).astype(np.int64))
                    doc_ids['begin_end'] = begin_end
                    if len(doc_ids['input_ids']) > 0:
                        self.data.append(doc_ids)
            # save cache
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data, f)                  
            end_time = time.time()
            print(f'Building dataset end, time cost: {end_time - start_time: .3f} s')

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tokens = torch.tensor(self.data[idx])
        item_data = {}
        for k, v in self.data[idx].items():
            item_data[k] = v.numpy()
        return item_data

def get_mask_and_position_ids(tokens, pad):
    batch_size, seq_length = tokens.shape

    # attention mask for flash-attn

    # attention_mask = tokens.ne(pad)
    # position_ids = torch.arange(seq_length, dtype=torch.long)
    # position_ids = position_ids.unsqueeze(0).expand_as(tokens)

    attention_mask = np.not_equal(tokens, pad)
    position_ids = np.arange(0, seq_length, dtype=np.int64) # [1, seq_len]
    position_ids = np.tile(position_ids, [batch_size, 1]) # [batch_size, seq_len]
    return attention_mask, position_ids

if __name__ == '__main__':
    root_folder = 'data'
    test_dataset = GPTJsonDataset(
        json_file=f'{root_folder}/web/refinedweb0.json',
        key='content',
        max_seq_len=1024,
        vocab_file=f'{root_folder}/vocab.json',
        merge_file=f'{root_folder}/merges.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    for idx, tokens in enumerate(test_dataloader):
        if idx > 4:
            break
        attention_mask, position_ids = get_mask_and_position_ids(tokens, test_dataset.encoder.pad_id())
        print(f'batch {idx}: shape = {tokens.shape}\ntokens = {tokens}\nattention_mask={attention_mask}\nposition_ids={position_ids}')