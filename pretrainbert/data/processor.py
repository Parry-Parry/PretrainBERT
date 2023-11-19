import os 
import re
import random
import torch
import logging
from nltk.tokenize import sent_tokenize

class StandardProcessor(object):
    def __init__(self, 
                 hf_dset, 
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

        self.hf_dset = hf_dset
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
    
    def _reset(self):
        self._current_sentences = []
        self._current_length = 0

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
        return self.hf_dset.map(
            function=self,
            batched=True if batch_size > 1 else False,
            remove_columns=self.hf_dset.column_names,
            disable_nullable=False,
            input_columns=self.columns,
            writer_batch_size=batch_size,
            num_proc=num_proc,
            **kwargs
        )

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
    
    def __call__(self, texts):
        dataset = {'input_ids':[], 'attention_mask' : [], 'segment_ids': [], 'nsp_label': [], 'mlm_positions': [], 'mlm_labels': []}
        for i, text in enumerate(texts): # for every doc
            lines = sent_tokenize(text)
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
                                random_document_index = random.randint(0, len(texts) - 1)
                                if random_document_index != i:
                                    break
                            if random_document_index == i:
                                label = 0
                            
                            random_document = texts[random_document_index]
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

                        dataset['input_ids'].append(tokens)
                        dataset['attention_mask'].append(attention_mask)    
                        dataset['segment_ids'].append(segment_ids)
                        dataset['nsp_label'].append(label)
                        dataset['mlm_positions'].append(masked_lm_positions)
                        dataset['mlm_labels'].append(masked_lm_labels)
                        self._reset()
        return dataset

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

    def __call__(self, texts, additionals):
        dataset = {'input_ids':[], 'attention_mask' : [], 'segment_ids': [], 'nsp_label': [], 'mlm_positions': [], 'mlm_labels': []}
        for i, text in enumerate(texts): # for every doc
            lines = sent_tokenize(text)
            second_segment = []
            j = 0
            while j < len(lines): # for every paragraph
                j += 1
                if len(self._current_sentences) == 0: # Just reset so find a new additional
                    label = 0
                    current_additional = additionals[i]
                    if random.random() < self._nsp_prob:
                        label = 1
                        for _ in range(10):
                            random_additional_index = random.randint(0, len(additionals) - 1)
                            if random_additional_index != i:
                                break
                        if random_additional_index == i:
                            label = 0
                        current_additional = additionals[random_additional_index]

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
                        for j in range(second_end):
                            second_segment.extend(self._current_sentences[j])
                        
                        second_segment = []
                        label = 1

                    if self.invert:
                        first_segment, second_segment = second_segment, first_segment
                        
                    self._truncate_seq_pair(first_segment, second_segment)
                    
                    tokens = [self.hf_tokenizer.cls_token_id] + first_segment + [self.hf_tokenizer.sep_token_id] + second_segment + [self.hf_tokenizer.sep_token_id]
                    segment_ids = [0] * (len(first_segment) + 2) + [1] * (len(second_segment) + 1) 

                    (tokens, masked_lm_positions, masked_lm_labels) = self._create_mlm(tokens)

                    attention_mask = [1] * len(tokens) + [0] * (self._max_length - len(tokens))
                    tokens = tokens + [self.hf_tokenizer.pad_token_id] * (self._max_length - len(tokens))

                    dataset['input_ids'].append(tokens)
                    dataset['attention_mask'].append(attention_mask)    
                    dataset['segment_ids'].append(segment_ids)
                    dataset['nsp_label'].append(label)
                    dataset['mlm_positions'].append(masked_lm_positions)
                    dataset['mlm_labels'].append(masked_lm_labels)
                    self._reset()

        return dataset