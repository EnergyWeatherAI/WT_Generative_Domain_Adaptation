import numpy as np
import pathlib, json, argparse
import torch

from data.scada_dataset import SCADA_Sample_Dataset
from models import base_models
from trainers import nbm_trainer
from evaluation import nbm_eval

#####################################
#   Main script to train a autoencoder-based normal behavior model for a single specified wind turbine.
#   train_NBM.py must be provided with WT information:
#   Example [python] train_NBM.py -SITE_NAME="farm1" -WT_ID=1 -SCARCITY="2w" 
#
#   Further settings are determined by the configuration dictionary within this script.
#   The script runs the corresponding training script and automatically saves the final NBM in the /saves/NBM/ folder.
######################################


# CLI parsing:
parser = argparse.ArgumentParser()
parser.add_argument('-SITE_NAME', help='wind site name, e.g., farm1')
parser.add_argument('-WT_ID', help='id of the WT, e.g., 1')
parser.add_argument('-CUDA_IDX', help='GPU CUDA index, exclude for cpu training', default = -1)
parser.add_argument('-SCARCITY', type=str, help='Set the scarcity scenario in (1w, 2w, 3w, 1m, 6w, 2m). \
                                            Do not provide anything to train with full training sets', default=None)
parser.add_argument('-VERBOSE', type=int, help='print out every verbose-th epoch', default=5)
parser.add_argument('-EPOCHS', type=int, help='how many epochs to train (at most)', default=5000)
parser.add_argument('-PATIENCE', type=int, help='early stopping patience (in epochs)', default=25)
args = parser.parse_args()


def main(args):
    np.random.seed(7)
    torch.manual_seed(7)

    device = torch.device(f'cuda:{args.CUDA_IDX}' if torch.cuda.is_available() else 'cpu')

    ################################
    # DATA SPECIFICATIONS & LOADING
    ################################
    SITE_NAME, WT_ID = args.SITE_NAME, args.WT_ID
    WT_NAME = f"{SITE_NAME}_WT_{WT_ID}"
    print(f"Preparing training NBM using WT: {WT_NAME} on device {device}")

    DATA_PATH = pathlib.Path.cwd().joinpath("dataset") # must contain the WT's raw SCADA .csv file
    scada_csv_path = DATA_PATH.joinpath(SITE_NAME, f"{SITE_NAME}_WT_{WT_ID}.csv")
    meta_csv_path = DATA_PATH.joinpath("META.csv")

    # save directory for the NBM model and the normalization statistics
    SCARCITY = "" if args.SCARCITY is None else f"_{args.SCARCITY}"
    save_dir = f"{WT_NAME}{SCARCITY}"
    pathlib.Path.cwd().joinpath("saves", "NBM", save_dir).mkdir(parents=True, exist_ok=True)

    # convert the scarcity degree (e.g., "2w") into number of SCADA sequences to include:
    period_to_scada = { "1w": 1008, # 10min = 1 val -> 6 per h ==> 6 * 24 = 144 per d => 144 * 7 =>  1008 SCADA vals in 1w
            "2w": 2016, "3w": 3024,"1m": 4032,"6w": 6048, "2m": 8064,"3m": 12096, "None": None}
    TR_LIMIT = None if args.SCARCITY is None else period_to_scada[args.SCARCITY]


    ###
    # *CONFIGURATION* to extract SCADA data 
    # Set here the SCADA features, sequence length, val/test split, and other settings
    ###
    config = {
        "SITE_NAME": SITE_NAME, # needed to extract site-specific rated wind speed and power

        "x_features": ["Power_min", "Power_avg", "Power_max", "WindSpeed_min", "WindSpeed_avg", "WindSpeed_max", 
                            "RotorSpeed_min", "RotorSpeed_avg", "RotorSpeed_max"] + ["StatorTemp1", "RotorTemp1"],
        
        "seq_len": 72, # 72 samples within a sequence <-> 12h
        "val_size": 0.30, # will be 30% of the (possibly artificially shortened) *training* set
        "test_size": 0.30, # will be the last 30% of data (i.e., is independent of the scarcity)
        "limit_tr_to": TR_LIMIT,  # how many samples to include in the training set (scarcity)
        "bs": 128, # batch size for training

        # Set whether to exclude incidents AND perform normal-data filtering to exclude outliers
        # Set to true for all training procedures.
        # NOTE: Would also affect the test set! Set to false when extracting a dataset for evaluation (see evaluate_models.py)
        "filter_incidents": True,
        "tr_shuffle": True # set to False only during evaluation to keep sequential order
    }

    # *DATASET EXTRACTION*
    # obtain SCADA data based on configuration, includes dataframes, np sequences, and torch datasets/dataloaders
    scada_ds = SCADA_Sample_Dataset(config, scada_csv_path, meta_csv_path).get_data()
    print(f"SCADA samples shape [scarcity check] (tr, val, test): {scada_ds["tr_samples"]["sequences"].shape}, {scada_ds["val_samples"]["sequences"].shape}, {scada_ds["test_samples"]["sequences"].shape}")

    # only the torch dataloaders with batches of data from the scada dataset are needed here
    tr_dl, val_dl, test_dl = scada_ds["torch_dataloaders"]

    # Store the normalization statistics in the save directory.
    # NOTE: Fine-tuning and domain mapping normalize (scarce) target WT data according to source WT statistics.
    # The statistics file calculated without data scarcity is there provided in the config to overwrite the normalization.
    # see train_domain_mapping.py and data/scada_dataset.py
    stats = scada_ds["tr_samples"]["stats"]
    stats_path = pathlib.Path.cwd().joinpath("saves", "NBM", save_dir, f"stats_{WT_NAME}.json")
    with open(stats_path, 'w+') as f: json.dump({k: list(stats[k]) for k in stats.keys()}, f, indent=4)



    ###############
    # NBM TRAINING
    ###############
    print("Starting training...")
    # initialize model & opt
    ae_model = base_models.base_AE_CNN(in_channels=len(config["x_features"])).to(device)
    opt = torch.optim.RAdam(ae_model.parameters(), lr=0.001)
    
    # initialize a trainer to perform the training loop
    ae_model_name = f"AE_model_{WT_NAME}.pt"
    model_save_path = pathlib.Path.cwd().joinpath("saves", "NBM", save_dir, ae_model_name)
    mytrainer = nbm_trainer.Trainer(opt=opt, verbose=args.VERBOSE, 
                                    es_patience=args.PATIENCE, device=device, model_save_path = model_save_path)

    # run training (logs/losses can be optionally saved)
    _logs = mytrainer.train(model=ae_model, epochs=args.EPOCHS, tr_dl=tr_dl, val_dl=val_dl)


    ###############
    # EVALUATION
    ###############

    # we perform an evaluation (calculating reconstruction errors) on normal-filtered data
    # the statistics are saved and are later used to determine the model's specific threshold (see evaluate_models.py)
    print(".... performing evaluation (on normal data only!)....")
    sets, dls = ["tr_N", "val_N"], [tr_dl, val_dl]
    results = {}
    for s, dl in zip(sets, dls):
        result_statistics = nbm_eval.get_result_statistics(ae_model, dl, device)
        results[s] = result_statistics
    
    results_save_path = pathlib.Path.cwd().joinpath("saves", "NBM", save_dir, "normal_data_results.json")
    with open(results_save_path, 'w+') as f: json.dump(results, f, indent=4)

    # clean up
    del ae_model
    if device !="cpu": torch.cuda.empty_cache()
    print("\n\n\n-----FINISHED---------")

if __name__ == "__main__":
    main(args)

