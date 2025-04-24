from torch import load, no_grad
from torch.nn import MSELoss
from utils.earlystop import EarlyStopper
from utils.loadsave import save_checkpoint
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class Trainer:

    '''
    This class is used to perform the actual training loops to train an autoencoder-based NBM. 
    train() can be called to fully train a NBM with a provided model and dataloaders, see train_NBM.py 
    '''

    def __init__(self, opt, verbose=False, es_patience = 15, device="cpu", model_save_path = None):
        '''
        Initializes a trainer instance with provided training settings. See train_NBM.py for an example.

        Args:
            opt (torch.optim): The already initialized optimizer to use for training.
            verbose (bool or int): Set to False for no console output. Otherwise it will output the progress every verbose-th epoch (set to 1 for all). 
            es_patience (int): Early stopping patience. Training will stop once es_patience epochs without progress in the criterion passed.
            device (torch device): The device to perform the training on (cuda-device or cpu).
            model_save_path (str): The training process will automatically save the model. Set the model save path (we use .pt filepaths).
        '''

        self.opt = opt
        self.es_patience = es_patience
        self.mse_fn = MSELoss()

        self.device = device
        self.model_save_path = model_save_path
        self.verbose = verbose

        # prepare for logging training progress
        self.logs = self._init_logs()


    def train(self, model, epochs, tr_dl, val_dl):

        '''
        Main training loop.

        Args:
            model (torch model): The initialized autoencoder-based NBM as torch model. 
            epochs (int): Maximum number of epochs to run - can be less with early stopping. 
            tr_dl | val_dl (torch dataloader): The training and validation torch dataloaders containing batches of data.
        '''
        
        model.to(self.device)

        # As specified in the paper appendix, we will use at evaluation the exponentional moving average weights of the model.
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.99)).to(self.device)

        # set up early stopping
        if self.es_patience is not None: es = EarlyStopper(patience=self.es_patience)


        # main training loop:
        for t in range(epochs):

            if self.verbose is not False: 
                print_progress = True if t % self.verbose == 0 else False
            else:
                print_progress = False 
                
            if print_progress: print(f"Epoch {t+1}...")
            
            # perform a training step
            self.train_step(tr_dl, model, ema_model)

            # perform a validation step (using the ema model)                                   
            self.validate_step(val_dl, ema_model)
            
            # progress report
            if print_progress:
                print(f"TR loss: {self.logs['tr'][-1]:.5f}, VAL loss: {self.logs['val'][-1]:.5f}")
                print("\n-------------------------------")

            # early stopping
            if self.es_patience is not None:
                best, stop = es.check_early_stop(self.logs["val"][-1])
                if best: save_checkpoint(self.model_save_path, ema_model.module, self.opt, t, self.logs["tr"][-1])
                if stop: break

        # loading best model checkpoint after training such that the model is at its best state
        print("Done!")
        if self.es_patience is not None:
            checkpoint = load(self.model_save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logs["best_val_loss"] = es.get_best_val_loss()
            print("Loaded (from EMA - weights!) model checkpoint from best epoch.")
        
        return self.logs # return the collected losses in logs for optional saving by the user


    def train_step(self, tr_dl, model, ema_model):
        '''
        Performs a training update step over all batches in the dataloader.

        Args:
            tr_dl: Training dataloader with training batches
            model: The torch model
            ema_model: The AveragedModel using EMA weights -- only used to update its weights during training 
        '''

        model.train()
        ema_model.train()

        epoch_loss = 0.
        num_batches = len(tr_dl)
        gen = iter(tr_dl)

        # standard training loop 
        for n_btch in range(num_batches):
            X = next(gen)
            X = X.to(self.device)

            # prep
            self.opt.zero_grad()

            # Compute prediction error
            pred = model(X)
            # prediction error (reconstruction mse_fn)
            mse = self.mse_fn(pred, X)
            # Backpropagation
            mse.backward()
            self.opt.step()

            # Update the EMA model parameters (see torch AveragedModel documentation)
            ema_model.update_parameters(model)

            # Gather data and report
            epoch_loss += mse.item()
    
        self._log({"tr": epoch_loss / num_batches})



    def validate_step(self, val_dl, ema_model):
        '''
        Performs a validation step over all validation batches using the model with EMA weights.

        Args:
            val_dl: The dataloader with validation data batches
            ema_model: The AveragedModel with EMA weights of the main NBM
        '''

        num_batches = len(val_dl)
        ema_model.eval()
        epoch_loss = 0.

        gen = iter(val_dl)

        # for each batch, collect the reconstruction error using the EMA (!) model
        with no_grad():
            for n in range(num_batches):
                X = next(gen)
                X = X.to(self.device)

                # Compute prediction error
                pred = ema_model(X)

                # prediction error
                mse = self.mse_fn(pred, X)
                epoch_loss += mse.item()

        self._log({"val": epoch_loss/num_batches})



    def _init_logs(self):
        logs = {} 
        logs["tr"] = []
        logs["val"] = []
        return logs 

    def _log(self, log_dict):
        for entry in log_dict:
            self.logs[entry].append(log_dict[entry])