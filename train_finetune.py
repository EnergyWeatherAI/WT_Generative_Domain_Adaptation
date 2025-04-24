import numpy as np
import pathlib, json, argparse, gc
import torch

from data.scada_dataset import SCADA_Sample_Dataset
from models import base_models
from trainers import nbm_trainer
from evaluation import nbm_eval


#####################################
#   Main script to finetune a pretrained source NBM to (scarce) target WT data. 
#   train_finetune.py must be provided with both domain's WT information:
#   Example [python] train_finetune.py -SITE_NAME_S="farm1" -WT_ID_S=1 -SITE_NAME_T="farm2" -WT_ID_T=1 -SCARCITY="2w" 
#
#   Further settings are determined by the configuration dictionary within this script.
#   The script runs the corresponding training script and automatically saves the finetuned NBM in the /saves/finetune/ folder.
######################################

# ---- CLI PARSING -----
parser = argparse.ArgumentParser()
parser.add_argument('-SITE_NAME_S', help='wind site name of the *source* WT, e.g., farm1')
parser.add_argument('-SITE_NAME_T', help='wind site name of the *target* WT, e.g., farm2')
parser.add_argument('-WT_ID_S', help='id of the source WT')
parser.add_argument('-WT_ID_T', help='id of the target WT')
parser.add_argument('-SCARCITY', type=str, help='Set the *target WT* scarcity scenario in (1w, 2w, 3w, 1m, 6w, 2m)', default="2w")
parser.add_argument('-CUDA_IDX', help='GPU CUDA index, exclude for cpu training', default = -1)
parser.add_argument('-VERBOSE', type=int, help='print out every verbose-th epoch', default=25)
parser.add_argument('-EPOCHS', type=int, help='how many epochs to train (at most)', default=5000)
parser.add_argument('-PATIENCE', type=int, help='early stopping patience (in epochs)', default=250)
args = parser.parse_args()

def main(args):

    print(f"FINETUNING source NBM of WT: {args.SITE_NAME_S}{args.WT_ID_S} using {args.SCARCITY} data of target WT: {args.SITE_NAME_T}{args.WT_ID_T}")

    np.random.seed(7)
    torch.manual_seed(7)
    device = torch.device(f'cuda:{args.CUDA_IDX}' if torch.cuda.is_available() else 'cpu')


    ################################
    # DATA SPECIFICATIONS & LOADING
    ################################
    DATA_PATH = pathlib.Path.cwd().joinpath("dataset") # must contain the WT's raw SCADA .csv file
    meta_csv_path = DATA_PATH.joinpath("META.csv")

    # SOURCE WT
    SITE_NAME_S, WT_ID_S = args.SITE_NAME_S, args.WT_ID_S
    csvpath_S = DATA_PATH.joinpath(SITE_NAME_S, f"{SITE_NAME_S}_WT_{WT_ID_S}.csv")
    WT_NAME_S = f"{SITE_NAME_S}_WT_{WT_ID_S}"

    # TARGET WT
    SITE_NAME_T, WT_ID_T = args.SITE_NAME_T, args.WT_ID_T
    csvpath_T = DATA_PATH.joinpath(SITE_NAME_T, f"{SITE_NAME_T}_WT_{WT_ID_T}.csv")
    WT_NAME_T = f"{SITE_NAME_T}_WT_{WT_ID_T}"
    TARGET_SCARCITY = args.SCARCITY

    # preparing saving
    save_dir_S = f"S_{WT_NAME_S}"
    save_dir_T = f"T_{WT_NAME_T}_{TARGET_SCARCITY}"
    pathlib.Path.cwd().joinpath("saves", "finetune", save_dir_S, save_dir_T).mkdir(parents=True, exist_ok=True)

    # convert the scarcity degree (e.g., "2w") into number of SCADA sequences to include:
    period_to_scada = { "1w": 1008, "2w": 2016, "3w": 3024,"1m": 4032, "6w": 6048, "2m": 8064,"3m": 12096, "None": None }
    TARGET_TR_LIMIT = None if args.SCARCITY is None else period_to_scada[args.SCARCITY]


    # NOTE: Fine-tuning and domain mapping normalize (scarce) target WT data according to (full dataset) source WT statistics.
    # Load the source WT's (representative) training statistics to normalize data according to those (see config_T)
    stats_path = pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_S[2:], f"stats_{WT_NAME_S}.json")
    with open(stats_path) as json_file: stats_S = json.load(json_file)


    ###
    # *CONFIGURATION* to extract SCADA data from the target WT
    # Set here the SCADA features, sequence length, val/test split, and other settings
    ###
    config_T = {
        "SITE_NAME": SITE_NAME_T, # needed to extract site-specific rated wind speed and power

        "x_features": ["Power_min", "Power_avg", "Power_max", "WindSpeed_min", "WindSpeed_avg", "WindSpeed_max", 
                            "RotorSpeed_min", "RotorSpeed_avg", "RotorSpeed_max"] + ["StatorTemp1", "RotorTemp1"],
        
        "seq_len": 72, # 72 samples within a sequence <-> 12h
        "val_size": 0.30, # will be 30% of the (possibly artificially shortened) *training* set
        "test_size": 0.30, # will be the last 30% of data (i.e., is independent of the scarcity)
        "limit_tr_to": TARGET_TR_LIMIT,  # how many samples to include in the training set (scarcity)
        "bs": 64, # batch size for training

        # Set whether to exclude incidents AND perform normal-data filtering to exclude outliers
        # Set to true for all training procedures.
        # NOTE: Would also affect the test set! Set to false when extracting a dataset for evaluation (see evaluate_models.py)
        "filter_incidents": True,
        "tr_shuffle": True, # set to False only during evaluation

        "overwrite_stats": stats_S, # supply source WT statistics to override normalization
    }


    # *DATASET EXTRACTION*
    # obtain SCADA data based on configuration, includes dataframes, np sequences, and torch datasets/dataloaders
    scada_ds_T = SCADA_Sample_Dataset(config_T, csvpath_T, meta_csv_path).get_data()
    print(f"SCADA sequences shapes [scarcity check] for target WT (tr, val, test): {scada_ds_T["tr_samples"]["sequences"].shape}, {scada_ds_T["val_samples"]["sequences"].shape}, {scada_ds_T["test_samples"]["sequences"].shape}")

    # only dataloaders/batches are required for finetuning
    tr_dl_T, val_dl_T, test_dl_T = scada_ds_T["torch_dataloaders"]


    ########################
    #     NBM FINETUNING  #
    ########################
    print("Starting fine-tuning...")

    # Initialize new ae-based NBM model
    ae_FT_model = base_models.base_AE_CNN(in_channels=len(config_T["x_features"]))
    
    # set weights to state of source domain NBM for fine-tuning
    src_NBM_name = f"AE_model_{WT_NAME_S}.pt"
    src_NBM_path = pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_S[2:], src_NBM_name)
    state_dict_S = torch.load(src_NBM_path)["model_state_dict"]
    ae_FT_model.load_state_dict(state_dict_S)
    ae_FT_model = ae_FT_model.to(device)

    # new save path (fine-tuned model)
    ae_FT_model_name = f"AE_model_S_{WT_NAME_S}_FT_T_{WT_NAME_T}_{TARGET_SCARCITY}.pt"
    ft_model_save_path = pathlib.Path.cwd().joinpath("saves", "finetune", save_dir_S, save_dir_T, ae_FT_model_name)


    ################
    # NBM FINETUNING
    ################
    # optimizer, set smaller learning rate
    opt = torch.optim.Adam([{"params": ae_FT_model.parameters()}], lr=0.0002)   
    # initialize trainer to perform the training loop
    mytrainer = nbm_trainer.Trainer(opt=opt, verbose=args.VERBOSE, es_patience=args.PATIENCE, device=device, model_save_path = ft_model_save_path)   

    # fine-tune
    _logs = mytrainer.train(model=ae_FT_model, epochs=args.EPOCHS, tr_dl=tr_dl_T, val_dl=val_dl_T)


    ###############
    # EVALUATION
    ###############

    # we perform an evaluation (calculating reconstruction errors) on normal-filtered data
    # the statistics are saved and are later used to determine the model's specific threshold (see evaluate_models.py)
    print(".... performing evaluation (on normal data only)....")
    sets, dls = ["tr_N", "val_N"], [tr_dl_T, val_dl_T]
    results = {}
    for s, dl in zip(sets, dls):
        result_statistics = nbm_eval.get_result_statistics(ae_FT_model, dl, device)
        results[s] = result_statistics
    results_save_path = pathlib.Path.cwd().joinpath("saves", "finetune", save_dir_S, save_dir_T, "normal_data_results.json")
    with open(results_save_path, 'w+') as f: json.dump(results, f, indent=4)

    # clean up
    del ae_FT_model
    gc.collect()
    if device !="cpu": torch.cuda.empty_cache()
    print("\n\n\n-----FINISHED---------")


if __name__ == "__main__":
    main(args)

