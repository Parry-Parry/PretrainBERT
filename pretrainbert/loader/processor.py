import os 
import random
import re
import torch
# Starting point from https://github.com/richarddwang/electra_pytorch/blob/master/_utils/utils.py#L101

class StdDataProcessor(object):
    def __init__(self, 
                 hf_dset, 
                 hf_tokenizer, 
                 max_length, 
                 text_col='text', 
                 lines_delimiter='\n', 
                 minimize_data_size=False, 
                 apply_cleaning=True,
                 prob : float = 0.15,
                 mask_prob : float = 0.85,
                 replace_prob : float = 0.1,
                 short_seq_prob : float = 0.1,
                 nsp_prob : float = 0.5,):
        
        self.hf_tokenizer = hf_tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._max_tokens = max_length - 3 # account for [CLS], [SEP], [SEP]
        self._target_length = max_length

        self.hf_dset = hf_dset
        self.text_col = text_col
        self.lines_delimiter = lines_delimiter
        self.minimize_data_size = minimize_data_size
        self.apply_cleaning = apply_cleaning

        self._prob = prob
        self._mask_prob = mask_prob
        self._replace_prob = replace_prob
        self._short_seq_prob = short_seq_prob
        self._nsp_prob = nsp_prob


    def mask_tokens(self, inputs):
        mask_token_index = self.hf_tokenizer.mask_token_id
        vocab_size = self.hf_tokenizer.vocab_size
        special_token_indices = [self.hf_tokenizer.cls_token_id, self.hf_tokenizer.sep_token_id]

        device = inputs.device
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self._prob, device=device)
        special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
        for sp_id in special_token_indices:
            special_tokens_mask = special_tokens_mask | (inputs == sp_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        mlm_mask = torch.bernoulli(probability_matrix).bool()
        labels[~mlm_mask] = -100  # We only compute loss on MLM applied tokens
    
        mask_token_mask = torch.bernoulli(torch.full(labels.shape, self._mask_prob, device=device)).bool() & mlm_mask
        inputs[mask_token_mask] = mask_token_index

        replace_token_mask = torch.bernoulli(torch.full(labels.shape, self._rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
        inputs[replace_token_mask] = random_words[replace_token_mask]

        return inputs, labels, mlm_mask

    def create_nsp_example(self, sentence1, corpus_sentences):
        input_ids = [self.hf_tokenizer.cls_token_id] + sentence1 + [self.hf_tokenizer.sep_token_id]
        sentA_length = len(input_ids)
        segment_ids = [0] * sentA_length

        max_num_tokens = self._max_length - 3
        target_seq_length = max_num_tokens

        if random.random() < self._short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        tokens_a = input_ids[1:-1]
        is_random_next = False

        if len(corpus_sentences) == 1 or random.random() < 0.5:
            is_random_next = True
            target_b_length = target_seq_length - len(tokens_a)

            for _ in range(10):
                random_document_index = random.randint(0, len(self.current_lines) - 1)

                if random_document_index != 0:
                    break

            random_document = self.current_lines[random_document_index]
            random_start = random.randint(0, len(random_document) - 1)
            tokens_b = random_document[random_start:]

            if len(tokens_b) > target_b_length:
                tokens_b = tokens_b[:target_b_length]
        else:
            is_random_next = False
            tokens_b = corpus_sentences

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)

        tokens, labels, mask = self.mask_tokens(tokens)

        return tokens, sentA_length, segment_ids, int(is_random_next), labels, mask

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

    def add_line(self, line):
        line = self.clean(line)
        tokens = self.hf_tokenizer.tokenize(line)
        tokids = self.hf_tokenizer.convert_tokens_to_ids(tokens)
        self._current_sentences.append(tokids)
        self._current_length += len(tokids)

        if self._current_length >= self._target_length:
            return self._create_example()

        return None

    def _create_example(self):
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            first_segment_target_length = (self._target_length - 3) // 2

        first_segment = []
        second_segment = []

        for sentence in self._current_sentences:
            if (len(first_segment) == 0 or
                    len(first_segment) + len(sentence) < first_segment_target_length or
                    (len(second_segment) == 0 and
                     len(first_segment) < first_segment_target_length and
                     random.random() < self._nsp_prob)):
                first_segment += sentence
            else:
                second_segment += sentence

        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(0, self._max_length - len(first_segment) - 3)]

        self._current_sentences = []
        self._current_length = 0

        if random.random() < self._short_seq_prob:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return self._make_example(first_segment, second_segment)

    def _make_example(self, first_segment, second_segment):
        input_ids, labels, mlm_mask = self.mask_tokens(first_segment)

        if random.random() < self._nsp_prob:
            input_ids, second_segment = second_segment, input_ids

        nsp_label = 0 if input_ids[0] == self.hf_tokenizer.cls_token_id else 1

        sentA_length = len(input_ids)
        segment_ids = [0] * sentA_length
        segment_ids += [1] * len(second_segment)

        if self.minimize_data_size:
            return {
                'input_ids': input_ids,
                'sentA_length': sentA_length,
                'nsp_label': nsp_label,
            }
        else:
            input_mask = [1] * len(input_ids)
            input_ids += [0] * (self._max_length - len(input_ids))
            input_mask += [0] * (self._max_length - len(input_mask))
            segment_ids += [0] * (self._max_length - len(segment_ids))
            labels += [-100] * (self._max_length - len(labels))
            mlm_mask += [False] * (self._max_length - len(mlm_mask))

            return {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'nsp_label': nsp_label,
                'mlm_labels': labels,
                'mlm_mask': mlm_mask,
            }
        
    def __call__(self, texts):
      new_example = {'input_ids':[], 'input_mask': [], 'segment_ids': [], 'nsp_label': [], 'mlm_labels': [], 'mlm_mask': []}
      for text in texts: # for every doc
        for line in re.split(self.lines_delimiter, text): # for every paragraph
          
          if re.fullmatch(r'\s*', line): continue # empty string or string with all space characters
          if self.apply_cleaning and self.filter_out(line): continue
          
          example = self.add_line(line)
          if example:
            for k,v in example.items(): new_example[k].append(v)
        
        if self._current_length != 0:
          example = self._create_example()
          for k,v in example.items(): new_example[k].append(v)

      return new_example
    
class CustomDataProcessor(StdDataProcessor):
    def __init__(self, 
                 hf_dset, 
                 hf_tokenizer, 
                 max_length, 
                 text_col='text', 
                 short_col='title',
                 lines_delimiter='\n', 
                 minimize_data_size=False, 
                 apply_cleaning=True, 
                 prob: float = 0.15, 
                 mask_prob: float = 0.85, 
                 replace_prob: float = 0.1, 
                 short_seq_prob: float = 0.1):
        super().__init__(hf_dset, hf_tokenizer, max_length, text_col, lines_delimiter, minimize_data_size, apply_cleaning, prob, mask_prob, replace_prob, short_seq_prob)

        self.short_col = short_col
      
    def map(self, **kwargs):
        num_proc = kwargs.pop('num_proc', os.cpu_count())
        return self.hf_dset.my_map(
            function=self,
            batched=True,
            remove_columns=self.hf_dset.column_names,
            disable_nullable=False,
            input_columns=[self.text_col, self.short_col],
            writer_batch_size=10**4,
            num_proc=num_proc,
            **kwargs
        )

    def add_line(self, line, title): # TODO: Add title to the line
        tokens = self.hf_tokenizer.tokenize(line)
        tokids = self.hf_tokenizer.convert_tokens_to_ids(tokens)
        self._current_sentences.append(tokids)
        self._current_length += len(tokids)

        if self._current_length >= self._target_length:
            return self._create_example()

        return None
    
    def __call__(self, texts, titles):
      new_example = {'input_ids':[], 'input_mask': [], 'segment_ids': [], 'nsp_label': [], 'mlm_labels': [], 'mlm_mask': []}
      for text, title in zip(texts, titles): # for every doc
        for line in re.split(self.lines_delimiter, text): # for every paragraph
          if re.fullmatch(r'\s*', line): continue # empty string or string with all space characters
          if self.apply_cleaning and self.filter_out(line): continue
          
          example = self.add_line(line, title)
          if example:
            for k,v in example.items(): new_example[k].append(v)
        
        if self._current_length != 0:
          example = self._create_example()
          for k,v in example.items(): new_example[k].append(v)

      return new_example

