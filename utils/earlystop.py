class EarlyStopper:
    '''
    Class responsible to keep track of the early stop criterion and return a stop flag when needed.
    '''
    def __init__(self, patience=1, min_delta=3e-5):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = 1000000
        self.min_delta = min_delta

    def check_early_stop(self, validation_loss):
        best, stop = False, False
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            best = True
        elif validation_loss >= self.min_validation_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: return False, True
        return best, stop

    def get_best_val_loss(self): return self.min_validation_loss