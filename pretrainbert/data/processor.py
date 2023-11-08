import os 
import re
import random
import torch

class StandardProcessor(object):
    def __init__(self, 
                 hf_dset, 
                 hf_tokenizer, 
                 max_length, 
                 text_col='text', 
                 lines_delimiter='\n', 
                 apply_cleaning=True,
                 short_seq_prob : float = 0.1,
                 nsp_prob : float = 0.5,
                 mlm_prob : float = 0.15,
                 original_prob : float = 0.1,
                 replace_prob : float = 0.1,
                 max_predictions_per_seq : int = 20):
        
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
        self._max_predictions_per_seq = max_predictions_per_seq

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

        device = tokens.device
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

        return tokens, mlm_mask, labels 

    def map(self, **kwargs):
        num_proc = kwargs.pop('num_proc', os.cpu_count())
        return self.hf_dset.my_map(
            function=self,
            batched=True,
            remove_columns=self.hf_dset.column_names,
            disable_nullable=False,
            input_columns=[self.text_col],
            writer_batch_size=10**4,
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
        dataset = {'input_ids':[], 'segment_ids': [], 'nsp_label': [], 'mlm_positions': [], 'mlm_labels': []}
        for i, text in enumerate(texts): # for every doc
            lines = re.split(self.lines_delimiter, text)
            for j, line in enumerate(lines): # for every paragraph
                if re.fullmatch(r'\s*', line): continue # empty string or string with all space characters
                if self.apply_cleaning and self.filter_out(line): continue

                self.add_line(line)

                if self._current_length >= self._target_length:
                    if self._current_sentences:
                        first_end = 1
                        if len(self._current_sentences) >= 2:
                            first_end = random.randint(1, len(self._current_sentences) - 1)
                        
                        first_segment = []
                        for j in range(first_end):
                            first_segment.extend(self._current_sentences[j])
                        
                        second_segment = []
                        label = 1

                        if len(self._current_sentences) == 1 or random.random() < self._nsp_prob:
                            label = 0

                            target_second_length = self._target_length - len(first_segment)
                            for _ in range(10):
                                random_document_index = random.randint(0, len(texts) - 1)
                                if random_document_index != i:
                                    break
                            if random_document_index == i:
                                label = 1
                            
                            random_document = texts[random_document_index]
                            random_document_lines = re.split(self.lines_delimiter, random_document)
                            random_document_tokids = self.process_document(random_document_lines)
                            random_start = random.randint(0, len(random_document_lines) - 1)

                            for j in range(random_start, len(random_document_tokids)):
                                second_segment.extend(random_document_tokids[j])
                                if len(second_segment) >= target_second_length:
                                    break
                            
                            num_unused_segments = len(self._current_sentences) - first_end
                            i -= num_unused_segments
                        else:
                            label = 1
                            for k in range(first_end, len(self._current_sentences)):
                                second_segment.extend(self._current_sentences[k])
                
                        self._truncate_seq_pair(first_segment, second_segment)
                        first_segment = [self.hf_tokenizer.cls_token_id] + first_segment + [self.hf_tokenizer.sep_token_id]
                        second_segment += [self.hf_tokenizer.sep_token_id]
                        segment_ids = self.hf_tokenizer.create_token_type_ids_from_sequences(first_segment, second_segment)
                        tokens = first_segment + second_segment
                        (tokens, masked_lm_positions, masked_lm_labels) = self._create_mlm(tokens)

                        dataset['input_ids'].append(tokens)
                        dataset['segment_ids'].append(segment_ids)
                        dataset['nsp_label'].append(label)
                        dataset['mlm_positions'].append(masked_lm_positions)
                        dataset['mlm_labels'].append(masked_lm_labels)
        return dataset

class CustomProcessor(StandardProcessor):
    def __init__(self, additional_col = 'title', **kwargs):
        super().__init__(**kwargs)
        self.additional_col = additional_col

    def map(self, **kwargs):
        num_proc = kwargs.pop('num_proc', os.cpu_count())
        return self.hf_dset.my_map(
            function=self,
            batched=True,
            remove_columns=self.hf_dset.column_names,
            disable_nullable=False,
            input_columns=[self.text_col, self.additional_col],
            writer_batch_size=10**4,
            num_proc=num_proc,
            **kwargs
        )
    
    def add_line(self, line):
        line = self.clean(line)
        tokens = self.hf_tokenizer.tokenize(line)
        tokids = self.hf_tokenizer.convert_tokens_to_ids(tokens)
        self._current_sentences.append(tokids)
        self._current_length += len(tokids)

        return None

    def __call__(self, texts, titles):
        dataset = {'input_ids':[], 'segment_ids': [], 'nsp_label': [], 'mlm_positions': [], 'mlm_labels': []}
        for i, (text, title) in enumerate(zip(texts, titles)): # for every doc
            lines = re.split(self.lines_delimiter, text)
            current_title = title
            if random.random() < self._nsp_prob:
                label = 0

                for _ in range(10):
                    random_title_index = random.randint(0, len(titles) - 1)
                    if random_title_index != i:
                        break
                if random_title_index == i:
                    label = 1
                
                current_title = titles[random_title_index]

            title_tokens = self.hf_tokenizer.tokenize(current_title)
            title_tokids = self.hf_tokenizer.convert_tokens_to_ids(title_tokens)
            first_segment = title_tokids
            self._current_length += len(title_tokids)

            second_segment = []

            for j, line in enumerate(lines): # for every paragraph
                if re.fullmatch(r'\s*', line): continue # empty string or string with all space characters
                if self.apply_cleaning and self.filter_out(line): continue

                self.add_line(line)

                if self._current_length >= self._target_length:
                    if self._current_sentences:
                        second_end = 1
                        if len(self._current_sentences) >= 2:
                            second_end = random.randint(1, len(self._current_sentences) - 1)
                        
                        second_segment = []
                        for j in range(second_end):
                            second_segment.extend(self._current_sentences[j])
                        
                        second_segment = []
                        label = 1
                
            self._truncate_seq_pair(first_segment, second_segment)
            
            tokens = [self.hf_tokenizer.cls_token_id] + first_segment + [self.hf_tokenizer.sep_token_id] + second_segment + [self.hf_tokenizer.sep_token_id]
            segment_ids = [0] * (len(first_segment) + 2) + [1] * (len(second_segment) + 1) 

            (tokens, masked_lm_positions, masked_lm_labels) = self._create_mlm(tokens)

            dataset['input_ids'].append(tokens)
            dataset['segment_ids'].append(segment_ids)
            dataset['nsp_label'].append(label)
            dataset['mlm_positions'].append(masked_lm_positions)
            dataset['mlm_labels'].append(masked_lm_labels)

        return dataset