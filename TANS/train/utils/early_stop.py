import numpy as np

class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_val = -np.inf
        self.best_dict = None
        self.early_stop = False

    def __call__(self, result):
        if result['val'] > self.best_val:
            self.best_val = result['val']
            self.best_dict = result
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop