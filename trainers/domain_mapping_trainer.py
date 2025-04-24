import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from utils.earlystop import EarlyStopper
from utils.loadsave import save_checkpoint
import itertools
from data import data_corrupter

class Trainer():
    
    '''
    This class is used to perform the actual training loops to train a domain mapping network. 
    train() can be called to fully train the network with a provided models and dataloaders, see train_domain_mapping.py 
    '''

    def __init__(self, config):
        '''
        Initializes a trainer instance with provided training settings. See train_domain_mapping.py for an example.

        Args:
            config (dict): A dictionary containing general training settings, with the following key/value expectations:
            ConfigDict:
                device (torch device): train the network on a cuda device or cpu
                save_dir (str): general save directory (folder) for both generator/mapping models
                es_patience (int): Early stopping patience. Training will stop once es_patience eval iterations without progress in the criterion passed.
                
                eval_model (torch model): The pretrained autoencoder-NBM of the source domain, used for early stopping criterion calculation.
                lambdas (dict): Hyperparameters, weights of the training losses
                
                max_power_S | max_power_T (float): maximum power of the source domain WT and the target domain WT (used for the rated power loss)
        '''

        self.device = config["device"]
        self.save_dir = config["save_dir"]
        self.es_patience = config["es_patience"]

        self.eval_model = config["evaluator_model"]
        self.lambdas = config["lambdas"]
        self.max_power_S, self.max_power_T = config["max_powers"]["S"], config["max_powers"]["T"]
        
        # L1 loss used in the consistency losses
        self.l1_loss = torch.nn.L1Loss()

        # The rated power loss is calculated in our study only for the mean power output (channel # 1), set the indices below
        self.power_ch = [1]
        # For the zero loss, the zero states are matched for the following channel indices:
        self.zero_ch = [0,1,2,6,7,8] # (our data: 0-2 power, 3-5 wind speed, 6-8 rotor speed)

    
        # prepare logging
        self.logs = self._init_logs(["GAN_ST", "GAN_TS", "dS", "dT", "gpST", "gpTS", 
                                            "cyc_s_t_s", "cyc_t_s_t", "zero_loss", "max_loss",
                                            "GAN_test_S", "GAN_tr_T_S", "GAN_tr_T_S_corr", "GAN_val_T_S", "GAN_val_T_S_corr", "GAN_test_T_S"])



    def train(self, gen_iter, models, dataloaders, optimizers, print_every = 100, evaluate_every = 100):

        '''
        Main training loop. See train_domain_mapping.py

        Args:
            gen_iter (int): Number of maximum *batch* iterations to perform - may be less with early stopping.
            models (dictionary): A dictionary containig two initialized generators (source->target, target->source) and two critics (source, target). 
            dataloaders (dictionary): Dictionary containing the dataloaders for batches of the training & validation sets of the source and target domain.
            optimizers (dictionary): Dictionary containing an optimizer for the generators and one for the discriminators.
            print_every (int): Prints out the progress in the console every print_every batch iteration
            evaluate_every (int): At every evaluate_every batch iteration, the losses and early stopping criterion are calculated. 
        '''
        # retrieve 4 models of the domain mapping network
        # genST refers to the generator mapping from Source to Target
        # discS refers to the discriminator evaluating samples for the source domain
        genST, genTS, discS, discT = [models[m] for m in ["genST", "genTS", "discS", "discT"]]
        
        # EMA for the generators (only) and only at evaluation time. See torch AveragedModel documentation
        genST_EMA = AveragedModel(genST, multi_avg_fn=get_ema_multi_avg_fn(0.99)).to(self.device)
        genTS_EMA = AveragedModel(genTS, multi_avg_fn=get_ema_multi_avg_fn(0.99)).to(self.device)
        
        # dataloaders
        tr_dl_S, tr_dl_T = [dataloaders[n] for n in ["tr_dl_S", "tr_dl_T"]] # _S refers to data from the source domain
        val_dl_S, val_dl_T = [dataloaders[n] for n in ["val_dl_S",  "val_dl_T"]]

        # optimizers
        opt_G = [optimizers[o] for o in ["opt_G"]][0]
        opt_D = [optimizers[o] for o in ["opt_D"]][0]

        # preparation
        self.print_every, self.evaluate_every = print_every, evaluate_every
        if self.es_patience is not None: self.es = EarlyStopper(patience=self.es_patience)
        #NOTE: The early stopping patience refers to the number of evaluations, not batch iterations. At evaluate_every = 100 and a patience of 10, this refers to 1000 iterations.


        ############
        # TRAINING #
        ############

        # use itertools to infinitely produce batches from the dataloader
        loaderS = itertools.cycle(tr_dl_S)
        loaderT = itertools.cycle(tr_dl_T)

        # for each batch iteration:
        for i in range(gen_iter+1):
            for model in [genST, genTS]: model.train()
            for model in [discS, discT]: model.eval()

            # 1) get a batch from the source domain and the target domain
            xs = next(loaderS).to(self.device)
            xt = next(loaderT).to(self.device)

            ##########################
            # FORWARD PASS GENERATOR #
            ##########################

            # map source -> target -> source (full cycle)
            S_mapped_to_T = genST(xs) # 'fake' target domain data
            S_cycled_back = genTS(S_mapped_to_T) # mapped back to original domain
            
            # equivalent for target -> source -> target
            T_mapped_to_S = genTS(xt)
            T_cycled_back = genST(T_mapped_to_S)

            ##########
            # LOSSES #
            ##########

            # GAN-QP LOSS
            loss_GAN_TS = torch.mean(discS(xs) - discS(T_mapped_to_S))
            loss_GAN_ST = torch.mean(discT(xt) - discT(S_mapped_to_T))
            gen_loss = loss_GAN_ST + loss_GAN_TS

            # CYCLE-CONSISTENCY LOSS
            loss_cyc_s_t_s = self.l1_loss(S_cycled_back, xs) 
            loss_cyc_t_s_t = self.l1_loss(T_cycled_back, xt)
            gen_loss = gen_loss + self.lambdas["cyc"] * (loss_cyc_s_t_s + loss_cyc_t_s_t)


            # ZERO-LOSS, see paper and _get_zero_loss
            if self.lambdas["zero"] is not None:
                zero_loss = self._get_zero_loss(real=xs, generated=S_mapped_to_T, channels=self.zero_ch)
                zero_loss = zero_loss + self._get_zero_loss(real=xt, generated=T_mapped_to_S, channels=self.zero_ch)
                gen_loss += (self.lambdas["zero"] * zero_loss)
                self._log({"zero_loss": zero_loss.item()})
            else:
                self._log({"zero_loss": 0.0})


            # RATED POWER LOSS, see paper and _get_rated_power_loss
            if self.lambdas["max"] is not None:
                max_loss = self._get_rated_power_loss(real=xt, generated=T_mapped_to_S,
                                                                rated_power_real = self.max_power_T, rated_power_gen = self.max_power_S, channels=self.power_ch)
                max_loss = max_loss + self._get_rated_power_loss(real=xs, generated=S_mapped_to_T,
                                                                rated_power_real = self.max_power_S, rated_power_gen = self.max_power_T, channels=self.power_ch)
                gen_loss += (self.lambdas["max"] * max_loss)
                self._log({"max_loss": max_loss.item()})
            else:
                self._log({"max_loss": 0.0})


            # ANOMALY AUGMENTATION PHASE (see paper appendix):
            # CYCLE LOSS using artificially corrupted data:

            # corrupt the original source and target domain data
            xsC = data_corrupter.corrupt_batch(xs) 
            xtC = data_corrupter.corrupt_batch(xt)

            # cycle a mapping to calculate the cycle-consistency loss
            SC_mapped_to_T = genST(xsC)
            SC_cycled_back = genTS(SC_mapped_to_T)
            
            # target -> source -> target
            TC_mapped_to_S = genTS(xtC)
            TC_cycled_back = genST(TC_mapped_to_S)
            
            loss_cyc_s_t_s_C = self.l1_loss(SC_cycled_back, xsC) 
            loss_cyc_t_s_t_C = self.l1_loss(TC_cycled_back, xtC)
            gen_loss = gen_loss + self.lambdas["cyc"] *  (loss_cyc_s_t_s_C + loss_cyc_t_s_t_C)



            ##########################
            # GENERATORS UPDATE STEP #
            ##########################
            opt_G.zero_grad()
            gen_loss.backward()
            opt_G.step()

            # EMA update
            genST_EMA.update_parameters(genST)
            genTS_EMA.update_parameters(genTS)


            ####################################
            #          CRITIC UPDATES          #
            ####################################
            for model in [genST, genTS]: model.eval()
            for model in [discS, discT]: model.train()
            for _ in range(1):
                # sample *new* batches
                xs = next(loaderS).to(self.device)
                xt = next(loaderT).to(self.device)

                with torch.no_grad(): # obtain mapped samples without updating generator
                    S_mapped_to_T = genST(xs)
                    T_mapped_to_S = genTS(xt)

                # update according to QP critic loss
                opt_D.zero_grad()
                disc_loss_S, disc_loss_T = self._critic_lossQP(discS, discT, T_mapped_to_S, S_mapped_to_T, xs, xt)
                disc_loss = disc_loss_S + disc_loss_T
                disc_loss.backward()
                opt_D.step()
            

            for model in [genTS, genST]: model.train()
            for model in [discS, discT]: model.eval()

            ###########
            # LOGGING #
            ###########
            self._log({"GAN_ST": loss_GAN_ST.item(), "GAN_TS": loss_GAN_TS.item(), "cyc_s_t_s": loss_cyc_s_t_s.item(), "cyc_t_s_t": loss_cyc_t_s_t.item(),
                        "dS": disc_loss_S.item(), "dT": disc_loss_T.item()})
            
            
            ############
            # PRINTING #
            ############
            if print_every is not None and i % print_every == 0:
                print("-----------------------------------------------")
                print("###############################################")
                print("-----------------------------------------------")
                print(f"Iteration {i}/{gen_iter}:")
                print(f"GAN (S->T): {self.logs['GAN_ST'][-1]:.2f} \t GAN (T->S): {self.logs['GAN_TS'][-1]:.2f}")
                print(f"DISC (S): {self.logs['dS'][-1]:.2f} \t DISC (T): {self.logs['dT'][-1]:.2f}")
                print(f"CYC (S->T->S): {self.logs['cyc_s_t_s'][-1]:.2f} \t CYC (T->S->T): {self.logs['cyc_t_s_t'][-1]:.2f}")
                print(f"ZERO loss: {self.logs['zero_loss'][-1]:.2f} \t MAX loss: {self.logs['max_loss'][-1]:.2f}")


            #################################
            # EVALUATION AND EARLY STOPPING #
            #################################
            if self.evaluate_every is not None:
                # get the mean reconstruction error of source val. data on the source NBM (eval model) as benchmark (not necessary)
                if i == 0: # only at the start
                    val_score_S_S = self.validate_S_on_S(val_dl_S)
                    self._log({"GAN_test_S": val_score_S_S})
                    if print_every is not None: 
                        print(f"Reconstruction Error S on NBM_S (Benchmark): \t \t Normal {val_score_S_S:.4f}")


                # evaluation AND early stopping every evaluate_every iteration
                if i % self.evaluate_every == 0: 
                    self.evaluate_model(genTS_EMA, genST_EMA, tr_dl_T, "tr") # tr data | use EMA (!) generator weights at evaluation
                    must_break = self.evaluate_model(genTS_EMA, genST_EMA, val_dl_T, "val") # get early stopping break flag for validation data
                    if must_break: break # early stop


        # training loop finish reached
        print(f"(reached iter limit @ {i})")

        # loading best model checkpoint (from EMA weights!) after training FINISHED
        if self.es_patience is not None:
            for model, savename in zip([genST, genTS], ["genST", "genTS"]):
                checkpoint = torch.load(self.save_dir.joinpath(f"{savename}.pt"), weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])

        return self.logs


    def _critic_lossQP(self, discS, discT, fakeS, fakeT, xs, xt):
        '''
        Calculates the discriminator loss for both critics. See paper and GAN-QP framework.

        Args:
            discS | discT: The discriminator of each domain
            fakeS: data mapped FROM the target domain TO the source domain, i.e., fake source domain data
            fakeT: fake target domain data, from mapping source data to the target data (genST(xt))
            xs | xt: real source | target domain batches
        '''

        # real data discriminator output
        f_xS = discS(xs)
        # discriminator output for generated data
        f_gS = discS(fakeS)

        # losses
        x_lossS = f_xS - f_gS
        x_normS = 1 * (xs - fakeS).abs().mean()
        x_lossS = -x_lossS + 0.5 * x_lossS ** 2 / x_normS
        x_lossS = x_lossS.mean()

        # real data discriminator output
        f_xT = discT(xt)
        # discriminator output for generated data
        f_gT = discT(fakeT)

        # losses
        x_lossT = f_xT - f_gT
        x_normT = 1 * (xt - fakeT).abs().mean()
        x_lossT = -x_lossT + 0.5 * x_lossT ** 2 / x_normT
        x_lossT = x_lossT.mean()        
        
        return x_lossS, x_lossT


    def _get_zero_loss(self, real, generated, channels = []):
        '''
        The zero loss, punishing zero states mapped to non-zero states.
        Due to our normalization adjustment, zero states in the power, windspeed, and rotor variables are normalized to -1.0
        '''

        # create a mask for positions where the provided REAL channels are at zero (roughly, from -0.99 to -1.01)
        mae = torch.tensor(0.0, device=self.device)
        for ch in channels:
            real_ch = real[:, ch, :]
            min_mask = torch.logical_and(real_ch <= -0.99, real_ch >= -1.01)
            # apply mask to fake/mapped/generated data to compare these positions only
            gen_ch = generated[:, ch, :]
            masked_gen = gen_ch[min_mask]
            # calculate mae loss on the masked data
            if min_mask.sum() > 0: mae += torch.mean(torch.abs(masked_gen - -1.0))
            else: mae += torch.tensor(0.0)
        return mae

    def _get_rated_power_loss(self, real, generated, rated_power_real, rated_power_gen, channels = []):
        '''
        Calculates the rated power loss, see paper. Punishes deviations between the rated power across domains. 
        '''

        # calculate a mask for where the power is at the rated capacity in the real sample
        # rated_power_real corresponds to the normalized value of the rated power for this WT
        real_ch = real[:, channels, :]
        max_mask = torch.logical_and(real_ch >= rated_power_real*0.99, real_ch <= rated_power_real * 1.01)
        
        # only consider those positions for the loss in the generated sample resembling the other domain
        gen_ch = generated[:, channels, :]
        masked_gen = gen_ch[max_mask]

        # calculate MAE loss
        # we expect the power to be at the (normalized) rated capacity of the other domain, rated_power_gen, and punish deviations from the generated value
        if max_mask.sum() > 0: mae = torch.mean(torch.abs(masked_gen - rated_power_gen))
        else: mae=torch.tensor(0.0)
        return mae


    def evaluate_model(self, genTS, genST, dl, dataset_name):   
        '''
        Evaluates the early stopping score for the domain mapping network in the target -> source domain direction.

        Args:
            genTS | genST: corresponding generator models. NOTE: Provide the EMA versions here if used.
            dl: dataloader with the batches from the target domain (typically).
            dataset_name (str): Used for logging and early stopping ("tr" or "val")
        '''         

        # calculate the early stopping criterion value for the corresponding dataset (tr/val)
        score_T_S = self.validate_T_to_S(genTS, dl, eval_batches = len(dl))
        self._log({f"GAN_{dataset_name}_T_S": score_T_S})
        # verbose output printing of the evaluated score 
        if self.print_every is not None:
            print(f"Reconstruction Error **T->S** on NBMs ({dataset_name}): \t \t {score_T_S:.4f}")

        # only apply early stopping w.r.t. validation score
        if dataset_name == "val":
            if self.es_patience is not None:
                best, stop = self.es.check_early_stop(score_T_S)
                if best: self.save_models(genTS, genST)
                if stop: return True 
                return False


    def validate_S_on_S(self, val_dl_S):
        '''
        Calculates the early stopping score (reconstruction error of normal data on the source NBM) using source data.
        This is only used to provide a benchmark value as to how the reconstruction is on real source data.
        '''
        val_data = iter(val_dl_S)
        running_score = 0.0
        with torch.no_grad():
            for m in range(len(val_dl_S)):
                realS = next(val_data)
                realS = realS.to(self.device)

                # EVALUATE ON EVAL MODEL
                reconstructions = self.eval_model(realS)
                running_score += self.l1_loss(realS, reconstructions)

        return running_score / len(val_dl_S)


    def validate_T_to_S(self, genT_to_S, val_dl_T, eval_batches = 10):
        '''
        Calculates early stopping score (NBM reconstruction error on source NBM/eval_model) with mapped target -> source data.
        '''
        genT_to_S.eval()
        val_data = iter(val_dl_T)
        running_score = 0.0

        # for each target batch:
        # map to source domain (genTS)
        # reconstruct using the eval_model (source NBM) and get avg reconstruction error (es score)
        with torch.no_grad():
            for m in range(eval_batches):

                realT= next(val_data)
                realT = realT.to(self.device)

                generation = genT_to_S(realT)

                reconstructions = self.eval_model(generation)
                running_score += self.l1_loss(generation, reconstructions)

        genT_to_S.train()
        return running_score / eval_batches


    def save_models(self, genTS, genST):
        if self.save_dir is None: print("NO MODELS SAVED - NO DIRECTORY SPECIFIED")
        else:
            names = ["genTS", "genST"]
            for name, model in zip(names, [genTS, genST]):
                # NOTE: Through model.module only the actual model is saved without the EMA/AveragedModel wrapper
                save_checkpoint(self.save_dir.joinpath(f"{name}.pt"), model.module, None, None, None)


    def _init_logs(self, keys):
        logs = {}
        for key in keys: logs[key] = []
        return logs

    def _log(self, log_dict):
        for entry in log_dict:
            self.logs[entry].append(log_dict[entry])