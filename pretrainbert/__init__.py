from enum import Enum
from .skeleton.models import ElectraForDiscrim, ElectraForNSP, ElectraForMultiObj
from .data.processor import StdProcessor, CustomProcessor
from .util import yaml_load

class ModelType(Enum):
    NSP = ElectraForNSP
    MultiObj = ElectraForMultiObj
    Discrim = ElectraForDiscrim

class ModelSize(Enum):
    small = {
        'mask_prob' : 0.15,
        'lr' : 5e-4,
        'bs' : 128,
        'steps' : 10**6,
        'max_length' : 128,
        'generator_size_divisor' : 4
    }
    base = {
        'mask_prob' : 0.15,
        'lr' : 2e-4,
        'bs' : 256,
        'steps' : 766*1000,
        'max_length' : 512,
        'generator_size_divisor' : 3
    }
    large = {
        'mask_prob' : 0.25,
        'lr' : 2e-4,
        'bs' : 2048,
        'steps' : 400*1000,
        'max_length' : 512,
        'generator_size_divisor' : 4
    }

def size_config(size):
    return ModelSize[size].value

def select_model(type):
    return ModelType[type].value