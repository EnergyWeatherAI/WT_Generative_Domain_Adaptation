import pandas as pd 
import numpy as np 
from data.csv_loader import load_csv_as_df
import data.torch_data as torch_data
import torch


class SCADA_Sample_Dataset:
    '''
    Class for handling the data sequencing and processing, and for retrieving datasets of one specific WT according to a configuration.

    Attributes for initialization:
        config (dict): A dictionary containing configuration settings for handling the dataset 
        scada_csv_path (str): The csv filepath corresponding to the WT's  
        meta_csv_path (str): The csv filepath corresponding to the meta file containing the rated wind speeds and power for all sites.


    This class handles the processing and retrieval of all forms of data for one WT.
    Its main method is get_data(), which returns a dictionary containing the extracted and processed data. Includes dataframes, normalized sequences, normalization functions, and torch data-sets/loaders. 
    Please refer to the method documentation.
    '''

    def __init__(self, config, scada_csv_path, meta_csv_path):
        '''
        Creates a dataset instance for the processing and retrieval of all forms of data for one WT.
        
        Args:
            config (dict): 
                    Expected config keys and values:
                        1) To specify the processing and sequencing of raw SCADA data:
                        x_features (list of str): A list of variable names to use as features within the sample, e.g. ["WindSpeed_min", "WindSpeed_avg", ...]
                        seq_len (int): The sequence length of a single SCADA sample. In our work, this is set to 72 timesteps (72 * 10 minutes = 12 hours) 
                        val_size (float): The validation size, as relative size of the training set (e.g., 0.30 for 30%)
                        test_size (float): The test set size, as relative size of the entire loaded SCADA dataframe (0.30 for 30%)
                        limit_tr_to (int or None): Sets the training data scarcity. Specifies how many training samples are to be included in the training set (e.g., 1008 for 1 week). Disregard or set to None to apply no data scarcity.
                        filter_incidents (bool): A flag to set whether normal data filtering and incident exclusion should be applied. Set to True for training datasets, False for evaluation datasets.

                        overwrite_stats (dict or None): By default, the data will be normalized according to training data statistics. In our work, we use the source WT statistics to normalize the to-be-mapped scarce target WT, for which we then provide a dictionary of statistics to use *instead*. 
                        site_name (str): The site name which is used to retrieve the meta information (rated windspeed and power), e.g., "farm1". Only used for meta retrieval.
            
                        2) Torch-specific configuration
                        bs (int): batch size for the torch dataloaders
                        tr_shuffle (bool): Whether to shuffle the training batches. Set to False during evaluation to retain the time-based order (i.e., sequential order over batches)

            scada_csv_path (str): The path to the raw SCADA .csv file to process
            meta_csv_path (str): The path to the meta .csv file which contains the rated power and wind speed per site

        See the training scripts (train_{model}.py) for use examples.

        '''

        self.config = config 
        self.scada_csv_path = scada_csv_path
        self.meta_csv_path = meta_csv_path


    def get_data(self):
        '''
        Returns a dictionary containing processed SCADA data according to the class config attributes.

        First, the loaded raw .csv file is converted to a dataframe and split into a training, validation, and test dataframe.
        If specified with the config['limit_tr_to'] setting, the training (and thus validation) dataframe are artificially made scarce. 
        Next, a sliding window approach is applied to obtain sequences (in our work: 12h sequences) as numpy arrays.
        Finally, the sequences are converted into torch datasets and dataloaders with a normalization function.
        '''
        
        # load the needed rated wind speed and rated power for filtering
        self.get_meta_data() 
        
        # load dataframe, split into tr/val/test, extract sequences (e.g., 12h sequences)
        tr_df, val_df, test_df, tr_data, val_data, test_data = self.extract_sequences() 

        # by default, obtain (de-)normalization function from the training_df 
        if self.config.get("overwrite_stats") is None: 
            norm_fX, unnorm_fX = self.get_normalization(tr_data["stats"])
        # otherwise, calculate the normalization functions using the provided statistics (e.g., using source WT statistics when normalizing scarce target WT data)
        else:
            norm_fX, unnorm_fX = self.get_normalization(self.config["overwrite_stats"])


        # convert into torch datasets and dataloaders, supply the (not yet applied) normalization function to torch
        tr_ds, val_ds, test_ds = self.get_torch_datasets(tr_data, val_data, test_data, norm_fX)
        tr_dl, val_dl, test_dl = self.get_torch_dataloaders(tr_ds, val_ds, test_ds)

        # extra: for the rated power loss (in domain mapping), we will need the rated power as a normalized value -> just apply normfX
        self.rated_pwr_normed = norm_fX(torch.broadcast_to(torch.Tensor([self.rated_pwr]), (1, len(self.config["x_features"]), self.config["seq_len"])))[0][1][0].item()

        # return results of all processing steps as dictionary
        scada_sample_dataset =  {
                "tr_df": tr_df, "val_df": val_df, "test_df": test_df, 
                "tr_samples": tr_data, "val_samples": val_data, "test_samples": test_data, 
                "torch_datasets": [tr_ds, val_ds, test_ds], 
                "torch_dataloaders": [tr_dl, val_dl, test_dl],
                "norm_fX": norm_fX, "unnorm_fX": unnorm_fX,
                "rated_ws": self.rated_ws, "rated_pwr": self.rated_pwr, "rated_pwr_normed": self.rated_pwr_normed,
            }

        return scada_sample_dataset
    

    def extract_sequences(self):
        '''Returns loaded and tr/val/test-split dataframes, and extracted sequences (e.g., 12h) obtained using a sliding window approach.'''
        
        # convert the csv file into a large dataframe
        wt_df = self.load_df()
        # split into training, validation, and test sets according to the configuration (sizes and scarcity degree)
        tr_df, val_df, test_df = self.split_df(wt_df)

        # split into sequences (tr_seq), with corresponding timestamps of the last value and incident_flags for each sequence
        tr_seq, tr_last_timestamps, tr_incident_flags  = self.split_into_sequences_extra(tr_df)
        # validation data
        val_seq, val_last_timestamps, val_incident_flags = self.split_into_sequences_extra(val_df)
        # test set data  
        test_seq, test_last_timestamps, test_incident_flags = self.split_into_sequences_extra(test_df)

        tr_data = {"sequences": tr_seq, "last_timestamps": tr_last_timestamps, "incident_flags": tr_incident_flags, "stats": self.adjust_statistics(self.get_stats(tr_df))}
        val_data = {"sequences": val_seq, "last_timestamps": val_last_timestamps, "incident_flags": val_incident_flags}
        test_data = {"sequences": test_seq, "last_timestamps": test_last_timestamps, "incident_flags": test_incident_flags}
        return tr_df, val_df, test_df, tr_data, val_data, test_data


    # split a dataframe into sequences
    def split_into_sequences_extra(self, df, convert_to_ch_first = True):
        '''
        Splits a provided dataframe into sequences using a sliding-window approach (e.g., 12h-samples)

        Args:
            df (pandas dataframe): The dataframe from which to extract sequences
            convert_to_ch_first (bool): By default, this procedure extracts sequences in the shape of datapoints (length) x features (channels). If true, convert to Torch-preferred ch x l
        '''
        # convert dataframe into a list, from then on a sliding window approach
        df_dict = df.to_dict(orient="list")
        x_seq = []
        last_timestamps, incident_flags = [], []

        # 1) for each row value n, extract the [n+seq_len] range.
        # 2) Only retain the specified features in the config x_features list
        # 3) Only add the features to the x_seq list if there is no NA value within the sequence
        # 4) Additionally, extract and retain the timestamp of the last value and whether there was an incident within the sequence (anywhere within e.g., 12h) 
        for row in range(len(df_dict["Timestamp"])-(self.config["seq_len"]+1)):
            # x features
            features = {}
            for f in self.config["x_features"]: 
                features[f] = df_dict[f][row:row+self.config["seq_len"]]
            
            x = np.dstack(([features[f] for f in self.config["x_features"]])).reshape(self.config["seq_len"], -1)
                
            if np.isnan(x).sum() == 0:
                x_seq.append(x)
                last_timestamps.append(df_dict["Timestamp"][row:row+self.config["seq_len"]][-1])
                incident_flags.append(np.array(df_dict["incident"][row:row+self.config["seq_len"]]).max())
                
        if convert_to_ch_first:  #highly recommended to keep this format due to other dependencies
            return np.moveaxis(np.asarray(x_seq), 1, 2), last_timestamps, np.asarray(incident_flags)
        else: 
            return np.asarray(x_seq), last_timestamps, np.asarray(incident_flags)



    def get_meta_data(self):
        '''Obtains the rated power and wind speed of the WT based on the meta csv file which stores information per site (assumption: all WTs identical within SITE)'''
        # see example META.csv file
        self.meta_df = pd.read_csv(self.meta_csv_path)    
        self.rated_pwr = float(self.meta_df[self.meta_df.SITE == self.config["SITE_NAME"]]["POWER"].item())
        self.rated_ws = float(self.meta_df[self.meta_df.SITE == self.config["SITE_NAME"]]["WINDSPEED"].item())



    def load_df(self, keep_incident_flag=True):
        # converts the SCADA csv into a dataframe, with filters applied, depending on settings. See csv_loader.
        return load_csv_as_df(self.scada_csv_path, self.config["x_features"], self.rated_ws, self.rated_pwr, self.config["filter_incidents"], keep_incident_flag) 


    def get_stats(self, tr_df):  
        '''Returns the minimum and maximum values of the dataframe (typically, training) for each x feature in a dictionary.'''
        def get_mins(df, cols): return df[cols].min(axis=0, numeric_only=True).to_numpy()
        def get_maxs(df, cols): return df[cols].max(axis=0, numeric_only=True).to_numpy()

        tr_X_mins = get_mins(tr_df, self.config["x_features"])
        tr_X_maxs = get_maxs(tr_df, self.config["x_features"])

        stats = {"X_mins": tr_X_mins, "X_maxs": tr_X_maxs}
        return stats


    def get_normalization(self, stats):
        '''
        Returns a torch-adapted transformation function to normalize (norm_fX) and de-normalize (unnorm_fX). To be provided for dataset instances.
        '''
        mins, maxs = torch.from_numpy(np.asarray(stats["X_mins"])).type(torch.float32), torch.from_numpy(np.asarray(stats["X_maxs"])).type(torch.float32)
        norm_fX = torch_data.get_normalize_f(mins, maxs, b=1, a=-1)
        unnorm_fX = torch_data.get_unnormalize_f(mins, maxs, b=1, a=-1)
        return norm_fX, unnorm_fX


    def adjust_statistics(self, stats):
        '''
        The statistics are calculated according to the minimum and maximum value. 
        However, for our zero consistency loss we prefer a way to quickly check for zero values in a normalized sample, which then would always have a different value.

        Therefore, we manually adjust the calculated training data statistics set such that they have a minimum value of 0.0 for the power, windspeed, and rotor speed.
        Temperature statistics are NOT adjusted, as they are not included in the zero consistency loss.
        
        NOTE: Assumes the specified order of x_features [3 power channels, 3 wind speed channels, 3 rotor channels, 2 temperature channels]. Adjust accordingly!
        '''
        #minimum: 0 for all power, windspeed, and rotorspeed values!
        stats["X_mins"] = np.array([0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0, stats["X_mins"][-2], stats["X_mins"][-1]])
        return stats





    def split_df(self, wt_df, verbose=False):
        '''
        Splits a WT SCADA dataframe into (possibly scarce) training, validation, and test dataframes (according to configuration) and returns them.
        '''

        # test size is INDEPENDENT of training (subset-) size. The last test_size*100% of data.
        test_df, n_test_set = self.get_test_df(wt_df)

        # if NO SCARCITY is set, just split the entire non-test remainder into a training and validation set.
        if self.config["limit_tr_to"] is None:
            train_data = wt_df[:-n_test_set]
            val_len = int(len(train_data) * self.config["val_size"])
            # split into a training set, which is again split into validation and an updated tr set
            val_df = train_data[-val_len:]
            tr_df = train_data[:-val_len] 
    
        else:
            # SCARCITY is selected by limiting the training data size (includes validation set) to config["limit_tr_to"] sequence samples! 
            # To have a continuous training and test transition, only the last -limit_tr_to sequences samples are retained (e.g., the last 2 weeks just BEFORE the test set)

            # we use find_idx_limit to find the appropriate cutoff to get exactly limit_tr_to samples (non-nan)
            # i.e., if there is no data in the week before the test set, it will go back in time until it collects enough samples.
            nontest = wt_df[:-n_test_set] 

            tr_entries = int(self.config["limit_tr_to"] * (1-self.config["val_size"]))
            val_entries = self.config["limit_tr_to"] - tr_entries 
            val_cutoff = self.find_idx_limit(nontest, val_entries)

            val_df = nontest[val_cutoff:]
            tr_cutoff = self.find_idx_limit(nontest[:val_cutoff], tr_entries)
            tr_df = nontest[tr_cutoff:val_cutoff]

        if verbose: print(f"Original DF lens: {len(tr_df)}, {len(val_df)}, {len(test_df)}") # print out dataframe sizes for checks
        return tr_df, val_df, test_df


    def get_test_df(self, wt_df):
        '''Returns a smaller dataframe of relative size test_size together with its size in rows from the dataframe representing the test set.'''
        n_test_set = int(self.config["test_size"] *len(wt_df)) 
        test_df = wt_df[-n_test_set:]
        return test_df, n_test_set

    def find_idx_limit(self, df, wanted_n):
        '''
        Given a dataframe with m rows, it goes backwards until it finds a row after which exactly wanted_n samples can be extracted.
        The dataframe[row:m] represents a dataframe from which exactly wanted_n samples can be extracted. 
        '''

        df_dict = df.to_dict(orient="list")
        n_consecutive_examples = 0

        # for each row (backwards, starting from the end), extract a sample.
        # If there is no NA value in the sequence, add to the counter.
        # Once the counter matches wanted_n, return the row.

        for row in range(len(df_dict["Timestamp"]), self.config["seq_len"]+1, -1):
            nas = 0
            cols = self.config["x_features"]
            for f in cols:
                features = df_dict[f][row-self.config["seq_len"]:row]
                nas += np.isnan(features).sum() 
            
            if nas == 0:
                n_consecutive_examples += 1
                if n_consecutive_examples > wanted_n:
                    break
        return row 




    def get_torch_datasets(self, tr_data, val_data, test_data, transform):
        '''
        Converts the extracted sequence samples into torch datasets (see torch_data.py).
        Args:
            tr_data (numpy arrays): np arrays of the training sequences (n_tr_sequences x sequence_shape)
            val_data (numpy arrays): for validation sequences
            test_data (numpy arrays): for test set sequences
            transform (torch transformation function): In our work, a normalization function adapated for torch tensors to normalizate an item from the dataset with.
        '''
        tr_ds, val_ds, test_ds = torch_data.get_torch_datasets([tr_data["sequences"], val_data["sequences"], test_data["sequences"]], transform = transform)
        return tr_ds, val_ds, test_ds


    def get_torch_dataloaders(self, tr_ds, val_ds, test_ds, drop_last = True):
        '''
        Converts the torch datasets into dataloaders with specific batch sizes and shuffle properties.
        Args:
            tr_ds | val_ds | test_ds (torch dataset): A torch dataset object , see get_torch_datasets and torch_data.py
            drop_last (bool): If set to true, the last batch is discarded. Needed for consistent and matching batch sizes in the mapping.
        '''
        tr_dl, val_dl, test_dl = torch_data.get_dataloaders([tr_ds, val_ds, test_ds], batch_sizes=[self.config["bs"], 128, self.config["bs"]], shuffles=[self.config["tr_shuffle"], False, False], drop_last = [drop_last, False, False]) 
        return tr_dl, val_dl, test_dl




