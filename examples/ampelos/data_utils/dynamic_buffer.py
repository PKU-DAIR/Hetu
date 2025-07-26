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

from .preprocessors import *
from multiprocessing.managers import BaseManager

class DatasetManager(BaseManager):
    pass

class DynamicBuffer():
    def __init__(self, buffersize):
        super(DynamicBuffer, self).__init__()
        self.buffersize = buffersize
        # self.fillsize = buffersize // 2
        self.fillsize = buffersize // 4
        self.total_len = 0
        self.tmp_consume = 0
        self.num_consume = 0
        self.buffer_ptr = 0
        self.buffer_map = {}
        initial_elements = [0] * self.buffersize
        self.buffer = deque(initial_elements, maxlen=self.buffersize)

    def fetch_data(self, num_items):
        raise NotImplementedError("Subclasses must implement this method")
    
    def next_ptr(self):
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffersize
    
    def already_eof(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def __len__(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement this method")

class DynamicGPTBuffer(DynamicBuffer):
    def __init__(self, json_file, key, max_seq_len, vocab_file, merge_file, 
                 buffersize=4096, skip_items=0):
        super(DynamicGPTBuffer, self).__init__(buffersize)
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
        self.max_seq_len = max_seq_len
        self.eof = False
        self.data = []
        self.lock = threading.Lock()
        self.file_lock = threading.Lock()

        # tokenize data from json file(without cache)

        # test read and skip time
        # test_items = 10000
        # st = time.time()
        # with open(json_file, 'r') as f_read:
        #     for i in range(test_items):
        #         json_line = f_read.readline() 
        #     json_line1 = f_read.readline() 
        # ed = time.time()
        # print("Read " + str(test_items) + " items,", "Cost:", ed - st,
        #       "Per Item:", (ed - st) / test_items)
        
        # st = time.time()
        # with open(json_file, 'r') as f_skip:
        #     for i in range(test_items):
        #         json_line = next(f_skip, None)
        #     json_line2 = f_skip.readline() 
        # ed = time.time()
        # print("Read " + str(test_items) + " items,", "Cost:", ed - st,
        #       "Per Item:", (ed - st) / test_items)
        # print("JSON1:", json_line1)
        # print("JSON2:", json_line2)
        # exit(0)

        self.open_file = open(json_file, 'r')
        self.skip_first_items(skip_items)
        self.total_len = skip_items
        self.fetch_data(self.buffersize)
        

    
    def drop_data(self, num_items, num_consume=None):
        st = time.time()
        if num_consume is None:
            num_consume = self.num_consume
        for i in range(num_items):
            if (len(self.buffer) > 0):
                # self.buffer.popleft()
                # print("DEL:", num_consume + i)
                del self.buffer_map[num_consume + i]
        ed = time.time()
        # print("DROP:", num_items, "TOTAL_LEN:", self.total_len,
        #       ", TIME COST:", ed - st, ", PER COST:", (ed - st) / num_items)

    def fetch_data(self, num_items):
        if self.eof:
            return 0
        # st = time.time()
        num_read = 0
        read_buffer = []
        self.file_lock.acquire_lock()
        st = time.time()
        read_str = "READ:" + str(self.total_len) + ", START:" + str(st)
        # print(read_str)
        while num_read < num_items:
            json_line = self.open_file.readline()
            if not json_line:
                print("End of file.")
                self.open_file.close()
                self.eof = True
                break
            # print("FOR ENCODE:", json_line)
            doc_ids = self.encoder.encode(json_line)
            if len(doc_ids) > 0:
                read_buffer.append(doc_ids)
                num_read += 1
        self.file_lock.release_lock()
        mid = time.time()
        read_str = "READ:" + str(self.total_len) + ", MID:" + str(mid - st) + "\n"
        # print(read_str)
        max_seq_len = self.max_seq_len + 1
        # self.lock.acquire_lock()
        for idx, doc_ids in enumerate(read_buffer):
            if len(doc_ids) > max_seq_len:
                self.buffer[self.buffer_ptr] = doc_ids[:max_seq_len]
                # self.buffer.append(doc_ids[:max_seq_len])
            elif len(doc_ids) < max_seq_len:
                self.buffer[self.buffer_ptr] = doc_ids + [self.encoder.pad_id()] * (max_seq_len - len(doc_ids))
                # self.buffer.append(doc_ids + [self.encoder.pad_id()] * (max_seq_len - len(doc_ids)))
            self.buffer_map[self.total_len] = self.buffer_ptr
            # print("INSERT:", self.total_len, "TO:", self.buffer_ptr)
            self.next_ptr()
            self.total_len += 1
        # self.lock.release_lock()
        ed = time.time()
        read_str = "READ:" + str(self.total_len - num_items) + ", END:" + str(ed)
        # print(read_str)
        read_str = "FETCH:" + str(num_items) + ", TOTAL_LEN:" + str(self.total_len) + \
                    ", TIME COST:" + str(ed - st) + ", PER COST:" + str((ed - st) / num_items)
        # print(read_str)
        return num_read
    
    def skip_first_items(self, num_items):
        if self.eof:
            return 0
        self.lock.acquire_lock()
        for i in range(num_items):
            next(self.open_file, None)
        self.lock.release_lock()
        return num_items

    def flush_buffer(self, num_consume):
        if (self.eof):
            return
        # print("Fill Data Num:", self.fillsize)
        self.lock.acquire_lock()
        self.drop_data(self.fillsize, num_consume)
        self.tmp_consume = 0
        self.lock.release_lock()
        self.fetch_data(self.fillsize)
    
    def get_len(self):
        return self.total_len
    
    def get_from_buffer(self, idx):
        self.lock.acquire_lock()
        tokens = np.array(self.buffer[idx])
        self.lock.release_lock()
        return tokens
    
    def get_item(self, idx):
        time0 = time.time()
        # read_str = "IDX:" + str(idx) + ", START:" + str(time0)
        # print(read_str)
        while(idx not in self.buffer_map):
            pass
        time1 = time.time()
        # read_str = "IDX:" + str(idx) + ", TIDX:" + str(self.buffer_map[idx]) + \
        #            ", Waiting for:" + str(time1 - time0)
        # print(read_str)
        self.lock.acquire_lock()
        tokens = np.array(self.buffer[self.buffer_map[idx]])
        self.lock.release_lock()
        return tokens

class DynamicMixBuffer(DynamicBuffer):
    def __init__(self, json_file, key, max_seq_len, vocab_file, merge_file, 
                 buffersize=4096, skip_items=0):
        super(DynamicMixBuffer, self).__init__(buffersize)
        # self.max_seq_len = max_seq_len
        self.eof = False
        self.data = []
        self.preprocessors = []
        self.cur_preprocessor = 0
        self.lock = threading.Lock()
        self.file_lock = threading.Lock()

        # just for debug
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
        test_encoder = GPT2BPEEncoder(args=args)
        test_processor = JsonPreProcessor(json_file, key, test_encoder, max_seq_len)
        self.preprocessors.append(test_processor)

        self.skip_first_items(skip_items)
        self.total_len = skip_items
        self.fetch_data(buffersize)
        

    
    def drop_data(self, num_items, num_consume=None):
        st = time.time()
        if num_consume is None:
            num_consume = self.num_consume
        for i in range(num_items):
            if (len(self.buffer) > 0):
                # self.buffer.popleft()
                # print("DEL:", num_consume + i)
                # print(f"drop {num_consume + i}")
                del self.buffer_map[num_consume + i]
        ed = time.time()
        # print("DROP:", num_items, "TOTAL_LEN:", self.total_len,
        #       ", TIME COST:", ed - st, ", PER COST:", (ed - st) / num_items)

    def fetch_data(self, num_items):
        if self.eof:
            return 0
        # st = time.time()
        remained_items = num_items
        read_buffer = []
        while (remained_items > 0):
            fetch_items, buffer = self.preprocessors[self.cur_preprocessor].fetch(remained_items)
            read_buffer.extend(buffer)
            remained_items -= fetch_items
            if remained_items > 0:
                self.cur_preprocessor += 1
                if self.cur_preprocessor == len(self.preprocessors):
                    break

        for idx, doc_ids in enumerate(read_buffer):
            # print(f"insert {self.total_len} to {self.buffer_ptr}")
            self.buffer[self.buffer_ptr] = doc_ids
            self.buffer_map[self.total_len] = self.buffer_ptr
            # print("FEEEEETCH:", self.total_len, ",", doc_ids)
            self.next_ptr()
            self.total_len += 1
        
        if remained_items > 0:
            self.eof = True
        return num_items - remained_items
    
    def skip_first_items(self, num_items):
        if self.eof:
            return 0
        remained_items = num_items
        for preprocessor in self.preprocessors:
            skip_items = preprocessor.skip(remained_items)
            remained_items -= skip_items
            if remained_items == 0:
                break
        print("SKIP:", skip_items)
        return num_items - remained_items

    def flush_buffer(self, num_consume):
        if (self.eof):
            return
        # print("Fill Data Num:", self.fillsize)
        self.lock.acquire_lock()
        self.drop_data(self.fillsize, num_consume)
        self.tmp_consume = 0
        self.lock.release_lock()
        self.fetch_data(self.fillsize)
    
    def already_eof(self):
        return self.eof
    
    def get_len(self):
        return self.total_len
    
    def get_from_buffer(self, idx):
        self.lock.acquire_lock()
        tokens = np.array(self.buffer[idx])
        self.lock.release_lock()
        return tokens
    
    def get_item(self, idx):
        time0 = time.time()
        # read_str = "IDX:" + str(idx) + ", START:" + str(time0)
        # print(read_str)
        while(idx not in self.buffer_map) and (not self.eof):
            pass
        time1 = time.time()
        # read_str = "IDX:" + str(idx) + ", TIDX:" + str(self.buffer_map[idx]) + \
        #            ", Waiting for:" + str(time1 - time0)
        # print(read_str)
        self.lock.acquire_lock()
        if (idx in self.buffer_map):
            tokens = np.array(self.buffer[self.buffer_map[idx]])
        else:
            tokens = None
        self.lock.release_lock()
        return tokens

DatasetManager.register("DGPTBuffer", DynamicGPTBuffer)
DatasetManager.register("DMixBuffer", DynamicMixBuffer)