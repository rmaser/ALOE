from pytorch_lightning.callbacks import Callback

class RecordNorms(Callback):
    '''
    Callback to record norms of the model for each module independently
    '''
    def __init__(self):
        super().__init__()
    
    
