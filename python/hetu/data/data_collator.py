import numpy as np
from typing import List, Dict, Any, Union
from hetu.data.tokenizers.pretrained_tokenizer import PreTrainedTokenizer
from dataclasses import dataclass
from hetu.data import IGNORE_INDEX

@dataclass
class DataCollatorForLanguageModel:
    """Data collator used for language model training.
    It will generate 'labels' if 'label_mask' is provided in the batch,
    otherwise it will replace pad_id of 'input_ids' with -100 for 'labels'.
    
    The returned batch will be a dictionary containing 'input_ids' and 'labels'.
    """
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, batch: List[Union[List[int], Dict[str, Any]]]) -> Dict[str, Any]:
        batch_fields = None
        if isinstance(batch[0], dict):
            batch_fields = batch[0].keys()
            if "input_ids" in batch[0] and "labels" in batch[0]:
                return batch

            if "input_ids" not in batch[0]:
                raise ValueError("batch should be a list of dictionaries with 'input_ids' key")
            input_ids = [example["input_ids"] for example in batch]
        else:
            input_ids = batch
        collate_batch = {"input_ids": [], "labels": []}
        for idx, doc_ids in enumerate(input_ids):
            labels = doc_ids[1:]
            doc_ids = doc_ids[:-1]
            if "label_mask" in batch_fields:
                label_mask = batch[idx]["label_mask"][1:]
                labels = np.where(label_mask == 1, IGNORE_INDEX, labels)
            else:
                # replace pad_id with -100 for labels
                labels = np.where(doc_ids == self.tokenizer.pad_id, IGNORE_INDEX, labels)
            collate_batch["input_ids"].append(doc_ids)
            collate_batch["labels"].append(labels)
        collate_batch["input_ids"] = np.array(collate_batch["input_ids"])
        collate_batch["labels"] = np.array(collate_batch["labels"])
        return collate_batch
