import os 
import os.path as path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from pretrainbert.util import mask_tokens

class DataPointer:
    def __init__(self, dir : str, raw : str = None) -> None:
        self.raw = raw if raw else dir
        self.dir = dir 
        self.init == False 

    def setup(self):
        self.init == True 

    def get_task_data(self, task):
        if path.exists(path.join(self.dir, task)):
            data = None
        else: 
            assert self.init, "Data Pointer must be initialized to pre-compute a task"
            os.makedirs(path.join(self.dir, task))
            data = None
        return data 


class CustomElectraDataset(Dataset):
    def __init__(self, tokenizer, text_data, max_seq_length, mlm_probability=0.15, replace_prob=0.1, original_prob=0.1):
        self.tokenizer = tokenizer
        self.text_data = text_data
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.replace_prob = replace_prob
        self.original_prob = original_prob

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]

        # Tokenize and truncate the text
        inputs = self.tokenizer(text, max_length=self.max_seq_length, truncation=True, padding="max_length", return_tensors="pt")

        # Create the MLM labels and attention mask using mask_tokens function
        mlm_input_ids, mlm_labels, mlm_mask = mask_tokens(
            inputs['input_ids'],
            mask_token_index=self.tokenizer.mask_token_id,
            vocab_size=len(self.tokenizer),
            special_token_indices=self.tokenizer.all_special_ids,
            mlm_probability=self.mlm_probability,
            replace_prob=self.replace_prob,
            orginal_prob=self.original_prob
        )

        # Create token_type_ids and attention_mask
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']

        # Create NSP labels
        if idx < len(self.text_data) - 1:
            # Check if there's a subsequent sentence
            next_text = self.text_data[idx + 1]
            is_next = True
        else:
            # If not, just repeat the same sentence (not a valid NSP pair)
            next_text = text
            is_next = False

        # Tokenize the next sentence
        next_inputs = self.tokenizer(next_text, max_length=self.max_seq_length, truncation=True, padding="max_length", return_tensors="pt")

        return {
            'input_ids': mlm_input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'mlm_labels': mlm_labels,
            'mlm_mask': mlm_mask,
            'nsp_labels': is_next,  # 1 if next sentence follows, 0 if not
        }