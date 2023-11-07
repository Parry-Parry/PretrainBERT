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
                 short_seq_prob : float = 0.1,):
        
        self.hf_tokenizer = hf_tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
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

    def __call__(self, texts):
        
        for text in texts:
            

