from fire import Fire 

# import model
from pretrainbert import select_model, size_config, yaml_load
from transformers import ElectraConfig, ElectraTokenizerFast

def main(config : str): 
    config = yaml_load(config)

    model_config = config['model']
    model_type = model_config.pop('type')
    model_size = model_config.pop('size')

    train_config = config['train']

    model_init = select_model(model_type)
    size_params = size_config(model_size)

    generator_config = ElectraConfig.from_pretrained(f'google/electra-{model_size}-generator')
    descriminator_config = ElectraConfig.from_pretrained(f'google/electra-{model_size}-discriminator')
    tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{model_size}-generator")

    if model_type != 'NSP': 
        generator_config.hidden_size = int(descriminator_config.hidden_size/size_params['generator_size_divisor'])
        generator_config.num_attention_heads =  int(descriminator_config.num_attention_heads//size_params['generator_size_divisor'])
        generator_config.intermediate_size =  int(descriminator_config.intermediate_size//size_params['generator_size_divisor'])

    model = model_init(
        generator_config_config=generator_config, 
        descriminator_config=descriminator_config, 
        tie_weights=train_config.pop('tie_weights', True)
        )
    
    test_data = [{
        'sentA' : "Hello, my dog is cute",
        'sentB' : "my cat is also cute"
    }]

    inputs = tokenizer(test_data, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    print(outputs)
    return "One pass test passed"

if __name__ == '__main__':
    Fire(main)
