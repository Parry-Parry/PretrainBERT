from fire import Fire 
import logging
from pretrainbert import StandardProcessor, CustomProcessor, yaml_load
from transformers import AutoTokenizer, ElectraTokenizerFast
from datasets import Dataset

def main(config : str):
    config = yaml_load(config)

    model_id = config.pop('model_id', 'google/electra-small-discriminator')
    process_type = config.pop('process_type', 'std')
    map_config = config.pop('map_config', {})

    dataset = config.pop('dataset')

    tokenizer = ElectraTokenizerFast.from_pretrained(model_id) if 'electra' in model_id else AutoTokenizer.from_pretrained(model_id)
    processor = StandardProcessor(dataset, tokenizer, **config) if process_type == 'std' else CustomProcessor(dset=dataset, hf_tokenizer=tokenizer, **config)
    logging.info(f"Using {processor.columns} columns")
    logging.info("Processing Dataset")
    dataset = processor.map(**map_config)
    return "Dataset saved Successfully"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)
