import os
import pandas as pd
import random
import torch
from nltk.tokenize import sent_tokenize
import ir_datasets as irds
from multiprocessing import Pool
from tqdm import tqdm
from more_itertools import chunked


class StandardProcessor(object):
    def __init__(self, 
                 dset, 
                 hf_tokenizer, 
                 max_length : int = 512, 
                 text_col='text', 
                 lines_delimiter='.', 
                 apply_cleaning=True,
                 short_seq_prob : float = 0.1,
                 nsp_prob : float = 0.5,
                 mlm_prob : float = 0.15,
                 original_prob : float = 0.1,
                 replace_prob : float = 0.1):
        
        self.hf_tokenizer = hf_tokenizer
        self.vocab = hf_tokenizer.get_vocab()
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length -3 

        self.irds = irds.load(dset)
        self.text_col = text_col
        self.lines_delimiter = lines_delimiter
        self.apply_cleaning = apply_cleaning

        self._short_seq_prob = short_seq_prob
        self._nsp_prob = nsp_prob
        self._mlm_prob = mlm_prob
        self._original_prob = original_prob
        self._replace_prob = replace_prob
        self._mask_prob = 1 - original_prob - replace_prob

        self.columns = [self.text_col]

        self.create_record = lambda inp, attn, seg, nsp, pos, labels: {'input_ids':inp, 'attention_mask':attn, 'segment_ids':seg, 'nsp_label':nsp, 'mlm_positions':pos, 'mlm_labels':labels}
    
    def _reset(self):
        self._current_sentences = []
        self._current_length = 0
    
    def _batch(self, iterable, bs):
        return chunked(iterable, bs)

    def _truncate_seq_pair(self, first_segment, second_segment):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(first_segment) + len(second_segment)
            if total_length <= self._max_length - 3:
                break

            trunc_tokens = first_segment if len(first_segment) > len(second_segment) else second_segment
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def _create_mlm(self, tokens):
        device = torch.device('cpu')
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        labels = tokens.clone()
        
        # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
        probability_matrix = torch.full(labels.shape, self._mlm_prob, device=device)
        special_tokens_mask = torch.full(tokens.shape, False, dtype=torch.bool, device=device)
        for sp_id in [self.hf_tokenizer.sep_token_id, self.hf_tokenizer.cls_token_id, self.hf_tokenizer.pad_token_id]:
            special_tokens_mask = special_tokens_mask | (tokens==sp_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        mlm_mask = torch.bernoulli(probability_matrix).bool()
        labels[~mlm_mask] = -100  # We only compute loss on mlm applied tokens

        # mask  (mlm_probability * (1-replace_prob-orginal_prob))
        mask_token_mask = torch.bernoulli(torch.full(labels.shape, self._mask_prob, device=device)).bool() & mlm_mask
        tokens[mask_token_mask] = self.hf_tokenizer.mask_token_id

        # replace with a random token (mlm_probability * replace_prob)
        if int(self._replace_prob)!=0:
            rep_prob = self._replace_prob/(self._replace_prob + self._orginal_prob)
            replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
            random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.long, device=device)
            tokens[replace_token_mask] = random_words[replace_token_mask]

        return tokens.numpy().tolist(), mlm_mask.numpy().tolist(), labels.numpy().tolist()

    def map(self, **kwargs):
        num_proc = kwargs.pop('num_proc', os.cpu_count())
        batch_size = kwargs.pop('batch_size', 10_000)
        batches = self._batch(self.irds.docs_iter(), batch_size)
        records = []
        from operator import length_hint

        def process_batch(batch):
            try:
                return self(batch)
            except Exception as e:
                print(f"Error processing batch: {e}")
                return []

        with Pool(num_proc) as pool, tqdm(total=length_hint(batches)) as pbar:
            for result in pool.imap(process_batch, batches):
                records.extend(result)
                pbar.update(1)

        return records

    def filter_out(self, line):
        return len(line) < 80

    def clean(self, line):
        return line.strip().replace("\n", " ").replace("()", "")

    def process_document(self, texts):
        tokens = [self.hf_tokenizer.tokenize(text) for text in texts]
        return [self.hf_tokenizer.convert_tokens_to_ids(token) for token in tokens]

    def add_line(self, line):
        line = self.clean(line)
        tokens = self.hf_tokenizer.tokenize(line)
        tokids = self.hf_tokenizer.convert_tokens_to_ids(tokens)
        self._current_sentences.append(tokids)
        self._current_length += len(tokids)

        return None
    
    def __call__(self, inputs):
        #dataset = {'input_ids':[], 'attention_mask' : [], 'segment_ids': [], 'nsp_label': [], 'mlm_positions': [], 'mlm_labels': []}
        batch = []
        inputs = pd.DataFrame(inputs)[self.columns]
        for i, row in enumerate(inputs.itertuples()): # for every doc
            lines = sent_tokenize(getattr(row, self.text_col))
            j = 0
            while j < len(lines): # while segments can exist
                try:
                    line = lines[j]
                except IndexError:
                    self._reset()
                    break

                self.add_line(line)
                j += 1

                if self._current_length >= self._target_length or j == len(lines) - 1: # Segments are ready
                    if self._current_sentences: # Sanity check
                        first_end = 1
                        if len(self._current_sentences) >= 2:
                            first_end = random.randint(1, len(self._current_sentences) - 1)
                        
                        first_segment = []
                        for k in range(first_end):
                            first_segment.extend(self._current_sentences[k])
                        
                        second_segment = []
                        label = 0

                        if len(self._current_sentences) == 1 or random.random() < self._nsp_prob: # NSP Swapping
                            label = 1

                            target_second_length = self._target_length - len(first_segment)
                            for _ in range(10):
                                random_document_index = random.randint(0, len(inputs) - 1)
                                if random_document_index != i:
                                    break
                            if random_document_index == i:
                                label = 0
                            
                            random_document = inputs.iloc[random_document_index][self.text_col]
                            random_document_lines = sent_tokenize(random_document)
                            random_document_tokids = self.process_document(random_document_lines)
                            if len(random_document_lines) == 0: 
                                label = 0
                                for k in range(first_end, len(self._current_sentences)):
                                    second_segment.extend(self._current_sentences[k])
                            else:
                                random_start = random.randint(0, len(random_document_lines) - 1)

                                for k in range(random_start, len(random_document_tokids)):
                                    second_segment.extend(random_document_tokids[k])
                                    if len(second_segment) >= target_second_length:
                                        break
                                num_unused_segments = len(self._current_sentences) - first_end # Reuse unused segments
                                j -= num_unused_segments # Reset iterator
                        else:
                            label = 0
                            for k in range(first_end, len(self._current_sentences)):
                                second_segment.extend(self._current_sentences[k])
                
                        self._truncate_seq_pair(first_segment, second_segment)
                        first_segment = [self.hf_tokenizer.cls_token_id] + first_segment + [self.hf_tokenizer.sep_token_id]
                        second_segment += [self.hf_tokenizer.sep_token_id]
                        segment_ids = self.hf_tokenizer.create_token_type_ids_from_sequences(first_segment, second_segment)
                        tokens = first_segment + second_segment

                        (tokens, masked_lm_positions, masked_lm_labels) = self._create_mlm(tokens)

                        attention_mask = [1] * len(tokens) + [0] * (self._max_length - len(tokens))
                        tokens = tokens + [self.hf_tokenizer.pad_token_id] * (self._max_length - len(tokens))

                        #dataset['input_ids'].append(tokens)
                        #dataset['attention_mask'].append(attention_mask)    
                        #dataset['segment_ids'].append(segment_ids)
                        #dataset['nsp_label'].append(label)
                        #dataset['mlm_positions'].append(masked_lm_positions)
                        #dataset['mlm_labels'].append(masked_lm_labels)
                        batch.append(self.create_record(tokens, attention_mask, segment_ids, label, masked_lm_positions, masked_lm_labels))
                        self._reset()
        return batch

class CustomProcessor(StandardProcessor):
    def __init__(self, additional_col = 'additional', invert=False, **kwargs):
        super().__init__(**kwargs)
        self.additional_col = additional_col
        self.invert = invert

        self.columns.append(self.additional_col)
    
    def add_line(self, line):
        line = self.clean(line)
        tokens = self.hf_tokenizer.tokenize(line)
        tokids = self.hf_tokenizer.convert_tokens_to_ids(tokens)
        self._current_sentences.append(tokids)
        self._current_length += len(tokids)

        return None

    def __call__(self, inputs):
        #dataset = {'input_ids':[], 'attention_mask' : [], 'segment_ids': [], 'nsp_label': [], 'mlm_positions': [], 'mlm_labels': []}
        batch = []
        inputs = pd.DataFrame(inputs)[self.columns]
        for i, row in enumerate(inputs.itertuples()): # for every doc
            lines = sent_tokenize(getattr(row, self.text_col))
            second_segment = []
            j = 0
            while j < len(lines): # for every paragraph
                j += 1
                if len(self._current_sentences) == 0: # Just reset so find a new additional
                    label = 0
                    current_additional = getattr(row, self.additional_col)
                    if random.random() < self._nsp_prob:
                        label = 1
                        for _ in range(10):
                            random_additional_index = random.randint(0, len(inputs) - 1)
                            if random_additional_index != i:
                                break
                        if random_additional_index == i:
                            label = 0
                        current_additional = inputs.iloc[random_additional_index][self.additional_col]

                    additional_tokens = self.hf_tokenizer.tokenize(current_additional)
                    additional_tokids = self.hf_tokenizer.convert_tokens_to_ids(additional_tokens)
                    first_segment = additional_tokids
                    self._current_length += len(additional_tokids)
                try:
                    line = lines[j]
                except IndexError:
                    self._reset()
                    break
                self.add_line(line)
                
                if self._current_length >= self._target_length or j == len(lines) - 1: # Segments are ready
                    if self._current_sentences:
                        second_end = 1
                        if len(self._current_sentences) >= 2:
                            second_end = random.randint(1, len(self._current_sentences) - 1)
                        
                        second_segment = []
                        for k in range(second_end):
                            second_segment.extend(self._current_sentences[k])
                        num_unused_segments = len(self._current_sentences) - second_end # Reuse unused segments
                        j -= num_unused_segments # Reset iterator

                    if self.invert:
                        first_segment, second_segment = second_segment, first_segment
                        
                    self._truncate_seq_pair(first_segment, second_segment)
                    
                    tokens = [self.hf_tokenizer.cls_token_id] + first_segment + [self.hf_tokenizer.sep_token_id] + second_segment + [self.hf_tokenizer.sep_token_id]
                    segment_ids = [0] * (len(first_segment) + 2) + [1] * (len(second_segment) + 1) 

                    (tokens, masked_lm_positions, masked_lm_labels) = self._create_mlm(tokens)

                    attention_mask = [1] * len(tokens) + [0] * (self._max_length - len(tokens))
                    tokens = tokens + [self.hf_tokenizer.pad_token_id] * (self._max_length - len(tokens))

                    #dataset['input_ids'].append(tokens)
                    #dataset['attention_mask'].append(attention_mask)    
                    #dataset['segment_ids'].append(segment_ids)
                    #dataset['nsp_label'].append(label)
                    #dataset['mlm_positions'].append(masked_lm_positions)
                    #dataset['mlm_labels'].append(masked_lm_labels)
                    batch.append(self.create_record(tokens, attention_mask, segment_ids, label, masked_lm_positions, masked_lm_labels))
                    self._reset()

        return batch