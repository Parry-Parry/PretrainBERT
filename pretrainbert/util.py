from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader 

def yaml_load(file : str):
    return load(open(file, 'r'), Loader=Loader)