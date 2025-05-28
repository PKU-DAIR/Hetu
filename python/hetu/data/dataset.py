import os
import json
import pickle
import time
import logging
import numpy as np
from typing import Optional
from tqdm import tqdm
from torch.utils.data import Dataset
from hetu.data.messages.message_template import MessageTemplate
from hetu.data.tokenizers.utils import BaseTokenizer
from hetu.data.tokenizers.pretrained_tokenizer import PreTrainedTokenizer

class JsonDataset(Dataset):
    """Dataset class for loading and processing JSON format text data
    
    This class inherits from torch.utils.data.Dataset, used for loading text data from JSON files,
    encoding text into token sequences, and unifying sequence lengths. Supports data caching to 
    improve loading speed.

    Args:
        json_file (str): Path to JSON data file
        text_field (str): JSON field name to extract text from
        tokenizer (BaseTokenizer): Tokenizer for encoding text
        max_seq_len (int): Maximum sequence length

    Example:
        >>> root_folder = 'data'
        >>> test_dataset = JsonDataset(
        ...     json_file=f'{root_folder}/web/refinedweb0.json',
        ...     text_field='content',
        ...     tokenizer=tokenizer,
        ...     max_seq_len=1024
        ... )
    """

    def __init__(
        self,
        json_file: str,
        text_field: str,
        tokenizer: BaseTokenizer,
        max_seq_len: int,
    ):
        self.tokenizer = tokenizer
        self.data = []

        cache_path = ''.join(json_file.split('.')[:-1]) + f'_cache.pkl'
        if os.path.exists(cache_path):
            # read exists data cache here
            logging.info(f'Loading exists cache from {cache_path} begin ...')
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            end_time = time.time()
            logging.info(f'Loading exists cache end, time cost: {end_time - start_time: .3f} s')
        else:
            # tokenize data from json file
            logging.info(f'Building dataset begin ...')
            start_time = time.time()
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                if isinstance(json_data, list):
                    for data in tqdm(json_data):
                        assert text_field in data, f'Field {text_field} not found in JSON data'
                        doc_ids = self.tokenizer.encode(data[text_field])
                        # doc_ids may be empty, will cause error when mbs=1
                        if len(doc_ids) > 0:
                            self.data.append(doc_ids)
                else:
                    assert text_field in json_data, f'Field {text_field} not found in JSON data'
                    doc_ids = self.tokenizer.encode(json_data[text_field])
                    self.data.append(doc_ids)
            except json.JSONDecodeError:
                # 如果不是标准json格式，可能是jsonl格式（每行一个json对象）
                with open(json_file, 'r') as f:
                    for json_line in tqdm(f):
                        data = json.loads(json_line.strip())
                        assert text_field in data, f'Field {text_field} not found in JSON data'
                        doc_ids = self.tokenizer.encode(data[text_field])
                        if len(doc_ids) > 0:
                            self.data.append(doc_ids)
            # save cache
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data, f)                    
            end_time = time.time()
            logging.info(f'Building dataset end, time cost: {end_time - start_time: .3f} s')

        # deal with max_seq_len + 1 (for tokens/labels seq_len = max_seq_len+1 - 1)
        logging.info(f'Truncating or padding data to max_seq_len + 1 = {max_seq_len + 1} begin ...')
        start_time = time.time()
        max_seq_len = max_seq_len + 1
        for idx, doc_ids in enumerate(self.data):
            if len(doc_ids) > max_seq_len:
                self.data[idx] = doc_ids[:max_seq_len]
            elif len(doc_ids) < max_seq_len:
                self.data[idx] += [self.tokenizer.pad_id] * (max_seq_len - len(doc_ids))
        end_time = time.time()
        logging.info(f'Truncating or padding data end, time cost: {end_time - start_time: .3f} s')

    def pad_id(self):
        return self.tokenizer.pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": np.array(self.data[idx])}

class SFTDataset(Dataset):
    """
    Dataset class for supervised fine-tuning with structured message format.
    
    This dataset handles loading and processing data for supervised fine-tuning,
    including message formatting and tokenization.
    
    Args:
        json_file (str): Path to JSON data file
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding text
        max_seq_len (int): Maximum sequence length
        message_template (MessageTemplate): Template for formatting messages
        prompt_template (Optional[str]): Template for formatting prompts
    """
    
    def __init__(
        self,
        json_file: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        message_template: MessageTemplate,
        prompt_template: Optional[str],
    ):
        self.tokenizer = tokenizer
        self.data = []

        cache_path = ''.join(json_file.split('.')[:-1]) + f'_cache.pkl'
        if os.path.exists(cache_path):
            # read exists data cache here
            logging.info(f'Loading exists cache from {cache_path} begin ...')
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            end_time = time.time()
            logging.info(f'Loading exists cache end, time cost: {end_time - start_time: .3f} s')
        else:
            # tokenize data from json file
            logging.info(f'Building dataset begin ...')
            start_time = time.time()
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                if isinstance(json_data, list):
                    for data in tqdm(json_data):
                        tokenized_message = tokenizer.tokenize_messages(
                            message_template(data),
                            prompt_template=prompt_template,
                        )
                        if len(tokenized_message["input_ids"]) > 0:
                            self.data.append(tokenized_message)
                else:
                    tokenized_message = tokenizer.tokenize_messages(
                        message_template(json_data),
                        prompt_template=prompt_template,
                    )
                    if len(tokenized_message["input_ids"]) > 0:
                        self.data.append(tokenized_message)
            except json.JSONDecodeError:
                # 如果不是标准json格式，可能是jsonl格式（每行一个json对象）
                with open(json_file, 'r') as f:
                    for json_line in tqdm(f):
                        data = json.loads(json_line.strip())
                        tokenized_message = tokenizer.tokenize_messages(
                            message_template(data),
                            prompt_template=prompt_template,
                        )
                        if len(tokenized_message["input_ids"]) > 0:
                            self.data.append(tokenized_message)
            # save cache
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data, f)                    
            end_time = time.time()
            logging.info(f'Building dataset end, time cost: {end_time - start_time: .3f} s')

        # deal with max_seq_len + 1 (for tokens/labels seq_len = max_seq_len+1 - 1)
        logging.info(f'Truncating or padding data to max_seq_len + 1 = {max_seq_len + 1} begin ...')
        start_time = time.time()
        max_seq_len = max_seq_len + 1
        for idx in range(len(self.data)):
            if len(self.data[idx]["input_ids"]) < max_seq_len:
                self.data[idx]["input_ids"] += [self.tokenizer.pad_id] * (max_seq_len - len(self.data[idx]["input_ids"]))
                self.data[idx]["label_mask"] += [1] * (max_seq_len - len(self.data[idx]["label_mask"]))
            else:
                self.data[idx]["input_ids"] = self.data[idx]["input_ids"][:max_seq_len]
                self.data[idx]["label_mask"] = self.data[idx]["label_mask"][:max_seq_len]
        end_time = time.time()
        logging.info(f'Truncating or padding data end, time cost: {end_time - start_time: .3f} s')

    def pad_id(self):
        return self.tokenizer.pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": np.array(self.data[idx]["input_ids"]),
            "label_mask": np.array(self.data[idx]["label_mask"])
        }
