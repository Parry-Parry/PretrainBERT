from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader 
from fire import Fire

from pretrainbert import BERTModel

def main(config : str): 
    config = load(open(config, 'r'), Loader=Loader)
    model_config = config['model']
    train_config = config['train']
    io_config = config['io']

    model = BERTModel.from_config(model_config)

    '''
    Initialization
    - Open Dataset 
    - Pre-Compute Objectives if needed
    - Optimizers and schedulers
    '''

    



if __name__ == '__main__': 
    Fire(main)