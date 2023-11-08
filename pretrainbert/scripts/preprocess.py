from fire import Fire 
from pretrainbert import StdProcessor, CustomProcessor, yaml_load
from transformers import AutoTokenizer

def main(config : str):
    config = yaml_load(config)

    model_id = config.pop('model_id', 'google/electra-small-discriminator')
    process_type = config.pop('process_type', 'std')
    map_config = config.pop('map_config', {})
    out_dir = config.pop('out_dir', './')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = StdProcessor(**config, hf_tokenizer=tokenizer) if process_type == 'std' else CustomProcessor(**config, hf_tokenizer=tokenizer)
    dataset = processor.map(**map_config)
    dataset.save_to_disk(out_dir)
    return "Dataset saved Successfully"

if __name__ == '__main__':
    Fire(main)
