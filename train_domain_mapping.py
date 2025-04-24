import numpy as np
import pathlib, json, argparse, gc
import torch

from data.scada_dataset import SCADA_Sample_Dataset
from models import domain_mapping_models
from trainers import domain_mapping_trainer
from utils.loadsave import load_pretrained_NBM


#####################################
#   Main script to train a domain mapping network with a specified source domain WT and target domain WT. 
#   train_domain_mapping.py must be provided with both domain's WT information:
#   Example [python] train_domain_mapping.py -SITE_NAME_S="farm1" -WT_ID_S=1 -SITE_NAME_T="farm2" -WT_ID_T=1 -SCARCITY="2w" 
#
#   Further settings are determined by the configuration dictionary within this script.
#   The script runs the corresponding training script and automatically saves the domain mappers (2 generators) in the /saves/mapping/ folder.
######################################

# ---- CLI PARSING -----
parser = argparse.ArgumentParser()
parser.add_argument('-SITE_NAME_S', help='wind site name of the *source* WT, e.g., farm1')
parser.add_argument('-SITE_NAME_T', help='wind site name of the *target* WT, e.g., farm2')
parser.add_argument('-WT_ID_S', help='id of the source WT')
parser.add_argument('-WT_ID_T', help='id of the target WT')
parser.add_argument('-SCARCITY', type=str, help='Set the *target WT* scarcity scenario in (1w, 2w, 3w, 1m, 6w, 2m)', default="2w")
parser.add_argument('-CUDA_IDX', help='GPU CUDA index, exclude for cpu training', default = -1)
parser.add_argument('-EVAL_EVERY', type=int, help='print out every verbose-th epoch', default=200)
parser.add_argument('-PATIENCE', type=int, help='early stopping patience (in evaluation steps)', default=5) # 20 = 10000 generator iterations
args = parser.parse_args()


def main(args):
    print(f"Training a domain mapping network with source WT: {args.SITE_NAME_S}{args.WT_ID_S} and target WT {args.SITE_NAME_T}{args.WT_ID_T} with target scarcity @ {args.SCARCITY}")

    np.random.seed(7)
    torch.manual_seed(7)
    device = torch.device(f'cuda:{args.CUDA_IDX}' if torch.cuda.is_available() else 'cpu')

    #torch.set_float32_matmul_precision("medium") # speeds up training (on compatible GPUs)

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
    save_dir_S, save_dir_T = f"S_{WT_NAME_S}", f"T_{WT_NAME_T}_{TARGET_SCARCITY}"
    pathlib.Path.cwd().joinpath("saves", "mapping", save_dir_S, save_dir_T).mkdir(parents=True, exist_ok=True)

    # convert the scarcity degree (e.g., "2w") into number of SCADA sequences to include:
    period_to_scada = { "1w": 1008, "2w": 2016, "3w": 3024,"1m": 4032, "6w": 6048, "2m": 8064,"3m": 12096, "None": None }
    TARGET_TR_LIMIT = None if args.SCARCITY is None else period_to_scada[args.SCARCITY]


    # NOTE: Fine-tuning and domain mapping normalize (scarce) target WT data according to (full dataset) source WT statistics.
    # Load the source WT's (representative) training statistics to normalize data according to those (see config_T)
    # For consistency, we also supply these saved statistics to the source domain to ensure shared normalization
    stats_path = pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_S[2:], f"stats_{WT_NAME_S}.json")
    with open(stats_path) as json_file: stats_S = json.load(json_file)

    ###
    # *CONFIGURATION* to extract SCADA data from the source & target WT
    # Apart from the data scarcity (only target WT) and the SITE_NAME, both share the same config
    # Set here the SCADA features, sequence length, val/test split, and other settings
    ###
    config_shared = {
        "x_features": ["Power_min", "Power_avg", "Power_max", "WindSpeed_min", "WindSpeed_avg", "WindSpeed_max", 
                            "RotorSpeed_min", "RotorSpeed_avg", "RotorSpeed_max"] + ["StatorTemp1", "RotorTemp1"],
        
        "seq_len": 72, # 72 samples within a sequence <-> 12h
        "val_size": 0.30, # will be 30% of the (possibly artificially shortened) *training* set
        "test_size": 0.30, # will be the last 30% of data (i.e., is independent of the scarcity)
        "bs": 128, # batch size for training

        # Set whether to exclude incidents AND perform normal-data filtering to exclude outliers
        # Set to true for all training procedures.
        # NOTE: Would also affect the test set! Set to false when extracting a dataset for evaluation (see evaluate_models.py)
        "filter_incidents": True,
        "tr_shuffle": True, # set to False only during evaluation

        "overwrite_stats": stats_S, # supply source WT statistics to override normalization
    }

    config_S = {"SITE_NAME": SITE_NAME_S, "limit_tr_to": None}
    config_S.update(config_shared)

    config_T = {"SITE_NAME": SITE_NAME_T, "limit_tr_to": TARGET_TR_LIMIT}
    config_T.update(config_shared)

    # load source WT data
    # obtain SCADA data based on configuration, includes dataframes, np sequences, and torch datasets/dataloaders
    scada_ds_S = SCADA_Sample_Dataset(config_S, csvpath_S, meta_csv_path).get_data()
    print(f"SCADA sequences shapes [scarcity check] for source WT (tr, val, test): {scada_ds_S["tr_samples"]["sequences"].shape}, {scada_ds_S["val_samples"]["sequences"].shape}, {scada_ds_S["test_samples"]["sequences"].shape}")

    scada_ds_T = SCADA_Sample_Dataset(config_T, csvpath_T, meta_csv_path).get_data()
    print(f"SCADA sequences shapes [scarcity check] for target WT (tr, val, test): {scada_ds_T["tr_samples"]["sequences"].shape}, {scada_ds_T["val_samples"]["sequences"].shape}, {scada_ds_T["test_samples"]["sequences"].shape}")

    # only dataloaders/batches are required for domain mapping
    tr_dl_S, val_dl_S, test_dl_S = scada_ds_S["torch_dataloaders"] # test dl will not be used
    tr_dl_T, val_dl_T, test_dl_T = scada_ds_T["torch_dataloaders"]


    ####################################
    #     DOMAIN MAPPING TRAINING      #
    ####################################
    print("Starting domain mapping training...")

    # Domain mapping components, 1 gan and discriminator for each domain
    discS = domain_mapping_models.DiscriminatorAE().to(device) # S -> discriminator for source domain
    discT = domain_mapping_models.DiscriminatorAE().to(device)
    genST = domain_mapping_models.GeneratorTCN().to(device) # ST -> maps Source to Target
    genTS = domain_mapping_models.GeneratorTCN().to(device)

    print("gen", sum(p.numel() for p in genST.parameters()))
    print("disc", sum(p.numel() for p in discT.parameters()))


    # optimizers
    opt_G =  torch.optim.Adam(list(genST.parameters()) + list(genTS.parameters()), lr=0.0002, betas=(0.5, 0.999))
    opt_D =  torch.optim.Adam(list(discS.parameters()) + list(discT.parameters()), lr=0.0002, betas=(0.5, 0.999))

    # for early stopping we need a pretrained source WT NBM:
    source_nbm_path = pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_S[2:], f"AE_model_{WT_NAME_S}.pt")
    source_nbm = load_pretrained_NBM(source_nbm_path, model_in_ch = len(config_S["x_features"]), device=device)

    # Prepare & initialize trainer to run the domain mapping network training loop
    save_dir_models = pathlib.Path.cwd().joinpath("saves", "mapping", save_dir_S, save_dir_T)
    lambdas = {"cyc": 30.0, "zero":.5, "max": .1}

    # trainer configuration (see trainers/domain_mapping_trainer)
    config_CG = {
        "lambdas": lambdas, # hyperparameters for loss weighting
        "evaluator_model": source_nbm,  # for early stopping criterion calculation (see paper appendix)
        "device": device, 
        "save_dir": save_dir_models, 
        "es_patience": args.PATIENCE,
        "max_powers" : {"S": scada_ds_S["rated_pwr_normed"], "T": scada_ds_T["rated_pwr_normed"]}, # for the rated power loss
        }

    mytrainer = domain_mapping_trainer.Trainer(config_CG)
    
    ###########
    # TRAINING#
    ###########
    mapping_network = {"genST": genST, "genTS": genTS, "discS": discS, "discT": discT}
    dataloaders = {"tr_dl_S": tr_dl_S, "tr_dl_T": tr_dl_T, "val_dl_S": val_dl_S, "val_dl_T": val_dl_T}
    opts = {"opt_G": opt_G, "opt_D": opt_D}

    _logs = mytrainer.train(gen_iter = 20001, models=mapping_network, dataloaders=dataloaders, optimizers=opts, print_every = args.EVAL_EVERY, evaluate_every = args.EVAL_EVERY)


    # clean up
    del genST, genTS, discS, discT
    gc.collect()
    if device !="cpu": torch.cuda.empty_cache()
    print("\n\n\n-----FINISHED---------")

if __name__ == "__main__":
    main(args)

