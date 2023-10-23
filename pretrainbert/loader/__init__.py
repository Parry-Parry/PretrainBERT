import os 
import os.path as path
from tqdm import tqdm

class DataPointer:
    def __init__(self, dir : str, raw : str = None) -> None:
        self.raw = raw if raw else dir
        self.dir = dir 
        self.init == False 

    def setup(self):
        self.init == True 

    def get_task_data(self, task):
        if path.exists(path.join(self.dir, task)):
            data = None
        else: 
            assert self.init, "Data Pointer must be initialized to pre-compute a task"
            os.makedirs(path.join(self.dir, task))
            data = None
        return data 


