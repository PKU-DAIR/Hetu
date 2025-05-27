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
from .dynamic_buffer import *


class DynamicDataset(Dataset):
    def __init__(self, buffersize):
        super(DynamicDataset, self).__init__()
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
        self.flush_start = multiprocessing.Array("i", [0], lock=True)
        self.flush_end = multiprocessing.Array("i", [0], lock=True)
        self.flush_num_consume = multiprocessing.Array("i", [0], lock=True)
        self.exit = multiprocessing.Array("i", [0], lock=True)
        self.flush_process = None
        self.eof = False

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

class DynamicJsonDataset(DynamicDataset):
    def __init__(self, json_file, key, max_seq_len, vocab_file, merge_file, 
                 buffersize=4096, skip_items=0):
        super(DynamicJsonDataset, self).__init__(buffersize)
        self.json_file = json_file
        self.key = key
        self.max_seq_len = max_seq_len
        self.vocab_file = vocab_file
        self.merge_file = merge_file
        self.skip_items = skip_items
        self.num_consume = skip_items
        self.manager_ctx = None
        self.start()
    
    def __del__(self):
        self.exit[0] = 1
    
    def start(self):
        if self.manager_ctx is not None:
            self.manager_ctx.shutdown()
            self.manager_ctx = None
        self.manager_ctx = DatasetManager()
        self.manager_ctx.start()
        self.dmanager = self.manager_ctx.DMixBuffer(self.json_file, self.key, self.max_seq_len, 
                                                    self.vocab_file, self.merge_file, 
                                                    self.buffersize, self.skip_items)
        
    def restart(self, skip_items):
        if self.manager_ctx is not None:
            self.manager_ctx.shutdown()
            self.manager_ctx = None
        self.skip_items = skip_items
        self.manager_ctx = DatasetManager()
        self.manager_ctx.start()
        self.dmanager = self.manager_ctx.DMixBuffer(self.json_file, self.key, self.max_seq_len, 
                                                    self.vocab_file, self.merge_file, self.buffersize, 
                                                    skip_items)

    
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
                # print("End of file.")
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
        num_skip = self.dmanager.skip_first_items(num_items)
        return num_skip

    def flush_buffer(self, num_consume):
        self.dmanager.flush_buffer(num_consume)
        # if (self.eof):
        #     return
        # print("Fill Data Num:", self.fillsize)
        # self.lock.acquire_lock()
        # self.drop_data(self.fillsize, num_consume)
        # self.tmp_consume = 0
        # self.lock.release_lock()
        # self.fetch_data(self.fillsize)
    
    def flush_buffer_round(self):
        while True:
            if (self.exit[0] == 1):
                break
            if (self.flush_start[0] == 0):
                time.sleep(0.5)
                continue
            if (self.flush_start[0] == 1):
                self.flush_start[0] = 0
                self.dmanager.flush_buffer(self.flush_num_consume[0])
                self.flush_end[0] = 0

    
    def already_eof(self):
        return self.dmanager.already_eof()
    
    def __len__(self):
        length = self.dmanager.get_len()
        return length
        # return self.total_len

    def __getitem__(self, idx):
        time0 = time.time()
        tokens = self.dmanager.get_item(idx)
        # tokens = self.dmanager.get_from_buffer(0)
        # self.lock.acquire_lock()
        self.tmp_consume += 1
        # print(f"idx:{idx}, self.num_consume:{self.num_consume}, fillsize:{self.fillsize}, buffersize:{self.buffersize}")
        if ((idx - self.num_consume) >= self.fillsize):
            # print(f"self.num_consume:{self.num_consume}, fillsize:{self.fillsize}")
            # p = threading.Thread(target=self.flush_buffer, args=(self.num_consume,))
            st = time.time()
            self.flush_start[0] = 1
            self.flush_end[0] = 0
            self.flush_num_consume[0] = self.num_consume
            if self.flush_process is None:
                self.flush_process = multiprocessing.Process(target=self.flush_buffer_round, 
                                                             args=())
                self.flush_process.start()
            # p = multiprocessing.Process(target=self.flush_buffer, args=(self.num_consume,))
            self.num_consume += self.fillsize
            # print("NEED FLUSH:", idx)
            # self.lock.release_lock()
            # p.start()
            ed = time.time()
            print(f"FLUSH_TIME:{ed - st}")
            # p.join()
            # self.flush_buffer(self.num_consume - self.fillsize)
        else:
            # print("DON'T NEED FLUSH:", idx)
            pass
        time1 = time.time()
        read_str = "IDX:" + str(idx) + ", Waiting for:" + str(time1 - time0)
        # print(read_str)
        return tokens
