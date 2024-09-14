from gensim.models.callbacks import CallbackAny2Vec
import time

class Word2VecProgress(CallbackAny2Vec):
    '''Callback to print loss after each epoch and track time.'''
    def __init__(self):
        self.epoch = 0
        self.start_time = time.time()

    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        print(f"Epoch #{self.epoch} end")
        print(f"Time taken for this epoch: {time.time() - self.start_time:.2f}s")
        self.epoch += 1
        self.start_time = time.time()