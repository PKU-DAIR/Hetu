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
from transformers import LlamaTokenizer

# Encoders
class BaseEncoder():
    def __init__(self):
        self.tokenizer = None

    def pad_id(self):
        raise NotImplementedError("Subclasses must implement this method")

    def encode(self, prompt):
        raise NotImplementedError("Subclasses must implement this method")

class Encoder(BaseEncoder):
    def __init__(self, args):
        self.args = args
        self.tokenizer = build_tokenizer(self.args)

    def pad_id(self):
        return self.tokenizer.pad

    def encode(self, prompt):
        data = json.loads(prompt)
        doc = data[self.args.key] # key: content for web, text for wiki
        assert (self.args.tokenizer_type == 'GPT2BPETokenizer'), 'Now only support GPT2BPETokenizer!'
        doc_ids = self.tokenizer.tokenize(doc)
        
        return doc_ids

class GPT2BPEEncoder(BaseEncoder):
    def __init__(self, args):
        self.args = args
        self.tokenizer = build_tokenizer(self.args)

    def pad_id(self):
        return self.tokenizer.pad

    def encode(self, prompt):
        doc_ids = self.tokenizer.tokenize(prompt)
        return doc_ids

class LlamaEncoder(BaseEncoder):
    def __init__(self, args):
        llama_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "tokenizer/models/llama2")
        print("LLAMAMODELPATH:", llama_dir)
        with open(os.path.join(llama_dir, "tokenizer_config.json"), 'r', encoding='utf-8') as file:  
            tokenizer_config = json.load(file)
        for k, v in tokenizer_config.items():  
            if isinstance(v, dict) and "__type" in v.keys() and v["__type"] == "AddedToken":
                v.pop("__type")
                tokenizer_config[k] = AddedToken(**v)
            
        self.tokenizer = LlamaTokenizer(vocab_file=os.path.join(llama_dir, "tokenizer.model"),
                                        **tokenizer_config)

    def pad_id(self):
        return self.tokenizer.eos_token_id

    def encode(self, prompt):
        doc_ids = self.tokenizer.encode(prompt)
        return doc_ids


#Text PreProcessors
class BasePreProcessor():
    def __init__(self):
        self.open_file = None
        self.file_lock = threading.Lock()
        self.eof = False

    def ready(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def skip(self, num_items):
        raise NotImplementedError("Subclasses must implement this method")
    
    def fetch(self, num_items):
        raise NotImplementedError("Subclasses must implement this method")

class JsonPreProcessor(BasePreProcessor):
    def __init__(self, filename: str, key: str, encoder: BaseEncoder, max_seq_len: str):
        super(JsonPreProcessor, self).__init__()
        self.filename = filename
        self.key = key
        self.encoder = encoder
        self.max_seq_len = max_seq_len
    
    def ready(self):
        self.open_file = open(self.filename, 'r')
    
    def skip(self, num_items):
        if self.open_file is None:
            self.ready()
        if self.eof:
            return 0
        self.file_lock.acquire_lock()
        for i in range(num_items):
            next(self.open_file, None)
        self.file_lock.release_lock()
        return num_items
        

    def fetch(self, num_items):
        if self.open_file is None:
            self.ready()
        num_read = 0
        read_buffer = []
        max_seq_len = self.max_seq_len + 1
        self.file_lock.acquire_lock()
        while num_read < num_items:
            json_line = self.open_file.readline()
            if not json_line:
                self.open_file.close()
                self.eof = True
                break
            prompt = json.loads(json_line)[self.key]
            doc_ids = self.encoder.encode(prompt)
            if len(doc_ids) > max_seq_len:
                doc_ids = doc_ids[:max_seq_len]
            elif len(doc_ids) < max_seq_len:
                doc_ids = doc_ids + [self.encoder.pad_id()] * (max_seq_len - len(doc_ids))
            if len(doc_ids) > 0:
                read_buffer.append(doc_ids)
                num_read += 1
        self.file_lock.release_lock()
        return num_read, read_buffer

class PlainTextPreProcessor(BasePreProcessor):
    def __init__(self, filename: str, encoder: BaseEncoder, max_seq_len: str):
        super(PlainTextPreProcessor, self).__init__()
        self.filename = filename
        self.encoder = encoder
        self.max_seq_len = max_seq_len
    
    def ready(self):
        self.open_file = open(self.filename, 'r')
    
    def skip(self, num_items):
        if self.open_file is None:
            self.ready()
        if self.eof:
            return 0
        self.file_lock.acquire_lock()
        for i in range(num_items):
            next(self.open_file, None)
        self.file_lock.release_lock()
        return num_items
        

    def fetch(self, num_items):
        if self.open_file is None:
            self.ready()
        num_read = 0
        read_buffer = []
        max_seq_len = self.max_seq_len + 1
        self.file_lock.acquire_lock()
        while num_read < num_items:
            json_line = self.open_file.readline()
            if not json_line:
                self.open_file.close()
                self.eof = True
                break
            doc_ids = self.encoder.encode(json_line)
            if len(doc_ids) > max_seq_len:
                doc_ids = doc_ids[:max_seq_len]
            elif len(doc_ids) < max_seq_len:
                doc_ids = doc_ids + [self.encoder.pad_id()] * (max_seq_len - len(doc_ids))
            if len(doc_ids) > 0:
                read_buffer.append(doc_ids)
                num_read += 1
        self.file_lock.release_lock()
        return num_read, read_buffer