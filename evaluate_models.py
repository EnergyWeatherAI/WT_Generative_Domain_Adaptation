import numpy as np
import pathlib, json, argparse
import pandas as pd
import torch

from data.scada_dataset import SCADA_Sample_Dataset
from evaluation import mapping_eval, nbm_eval
from utils.loadsave import load_pretrained_NBM, load_pretrained_mapper

#####################################
#   Evaluation script for pretrained standard NBMs, scarce NBMs, finetuned NBMs, and the domain mapping approach.
#   evaluate_models.py must be provided with both domain's WT information:
#   Example [python] evaluate_models.py -SITE_NAME_S="farm1" -WT_ID_S=1 -SITE_NAME_T="farm2" -WT_ID_T=1 -SCARCITY="2w" 
#
#   The script evaluates the same test set of the target WT, independent of the scarcity degree.
#   Further settings are determined by the configuration dictionary within this script.
#   The script stores the calculated results.csv in the respective results folder. 
######################################

# ---- CLI PARSING -----
parser = argparse.ArgumentParser()
parser.add_argument('-SITE_NAME_S', help='wind site name of the *source* WT, e.g., farm1')
parser.add_argument('-SITE_NAME_T', help='wind site name of the *target* WT, e.g., farm2')
parser.add_argument('-WT_ID_S', help='id of the source WT', default=1)
parser.add_argument('-WT_ID_T', help='id of the target WT', default=1)
parser.add_argument('-SCARCITY', type=str, help='Set the *target WT* scarcity scenario in (1w, 2w, 3w, 1m, 6w, 2m)', default="2w")
parser.add_argument('-CUDA_IDX', help='GPU CUDA index, exclude for cpu training', default = -1)
parser.add_argument('-NO_SCARCE', default=False, action=argparse.BooleanOptionalAction, help="Exclude scarce NBM evaluation")
parser.add_argument('-NO_FINETUNE', default=False, action=argparse.BooleanOptionalAction, help="Exclude evaluating finetuned models")
parser.add_argument('-NO_DOMAINMAPPING', default=False, action=argparse.BooleanOptionalAction, help="Set to exclude domain mapping approach from evaluation")
args = parser.parse_args()

def main(args):
    np.random.seed(7)
    torch.manual_seed(7)
    device = torch.device(f'cuda:{args.CUDA_IDX}' if torch.cuda.is_available() else 'cpu')

    ################################
    # DATA SPECIFICATIONS & LOADING
    ################################
    DATA_PATH = pathlib.Path.cwd().joinpath("dataset") # must contain the WT's raw SCADA .csv file
    meta_csv_path = DATA_PATH.joinpath("META.csv")

    # SOURCE WT information
    # we do not need any data from the source domain, but still identifier information
    SITE_NAME_S, WT_ID_S = args.SITE_NAME_S, args.WT_ID_S
    WT_NAME_S = f"{SITE_NAME_S}_WT_{WT_ID_S}"

    # TARGET WT
    SITE_NAME_T, WT_ID_T = args.SITE_NAME_T, args.WT_ID_T
    csvpath_T = DATA_PATH.joinpath(SITE_NAME_T, f"{SITE_NAME_T}_WT_{WT_ID_T}.csv")
    WT_NAME_T = f"{SITE_NAME_T}_WT_{WT_ID_T}"
    TARGET_SCARCITY = args.SCARCITY

    # convert the scarcity degree (e.g., "2w") into number of SCADA sequences to include:
    period_to_scada = { "1w": 1008, "2w": 2016, "3w": 3024,"1m": 4032, "6w": 6048, "2m": 8064,"3m": 12096, "None": None }
    TARGET_TR_LIMIT = None if args.SCARCITY is None else period_to_scada[args.SCARCITY]

    # preparing loading
    save_dir_S = f"S_{WT_NAME_S}"
    save_dir_T = f"T_{WT_NAME_T}_{TARGET_SCARCITY}"



    ###
    # *CONFIGURATION* to extract SCADA data from the target WT
    # Set here the SCADA features, sequence length, val/test split, and other settings
    ###

    config_base = {
        "SITE_NAME": SITE_NAME_T, # needed to extract site-specific rated wind speed and power

        "x_features": ["Power_min", "Power_avg", "Power_max", "WindSpeed_min", "WindSpeed_avg", "WindSpeed_max", 
                            "RotorSpeed_min", "RotorSpeed_avg", "RotorSpeed_max"] + ["StatorTemp1", "RotorTemp1"],
        
        "seq_len": 72, # 72 samples within a sequence <-> 12h
        "val_size": 0.30, # will be 30% of the (possibly artificially shortened) *training* set
        "test_size": 0.30, # will be the last 30% of data (i.e., is independent of the scarcity)
        "limit_tr_to": TARGET_TR_LIMIT,  # how many samples to include in the training set (scarcity)
        "bs": 64, # batch size for evaluation

        # NOTE: At evaluation time, we do NOT exclude incidents or outliers!
        "filter_incidents": False,
        "tr_shuffle": False, # set to False to keep order of batches sequential
    }


    ################################
    # DATA SPECIFICATIONS & LOADING
    ################################

    # NOTE: The representative target NBM was normalized according to the statistics of the *full* target WT training set 
    # set statistics accordingly in the configuration
    with open(pathlib.Path.cwd().joinpath("saves", "NBM", WT_NAME_T, f"stats_{WT_NAME_T}.json")) as json_file: 
        stats_full_target = json.load(json_file)
    config_normed_by_target = {"overwrite_stats": stats_full_target}
    config_normed_by_target.update(config_base)

    # get the corresponding test set dataloader with this normalization scheme:
    scada_ds_norm_full = SCADA_Sample_Dataset(config_normed_by_target, csvpath_T, meta_csv_path).get_data()
    _, _, test_dl_representativeNBM = scada_ds_norm_full["torch_dataloaders"] # only care about test dl



    # NOTE: The *scarce* target NBM was normalized according to the statistics of the *scarce* target WT training set 
    if not args.NO_SCARCE:
        with open(pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_T[2:], f"stats_{WT_NAME_T}.json")) as json_file: 
            stats_scarce = json.load(json_file)

        config_normed_by_scarce = {"overwrite_stats": stats_scarce}
        config_normed_by_scarce.update(config_base)
        scada_ds_norm_scarce = SCADA_Sample_Dataset(config_normed_by_scarce, csvpath_T, meta_csv_path).get_data()
        _, _, test_dl_scarce = scada_ds_norm_scarce["torch_dataloaders"]

    # NOTE: The fine-tune and domain mapping approach use statistics from the *source* WT to normalize the data 
    if not args.NO_FINETUNE or not args.NO_DOMAINMAPPING:
        with open(pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_S[2:], f"stats_{WT_NAME_S}.json")) as json_file: 
            stats_S = json.load(json_file)
 
        config_normed_by_source = {"overwrite_stats": stats_S}
        config_normed_by_source.update(config_base)
        scada_ds_norm_ft_dm = SCADA_Sample_Dataset(config_normed_by_source, csvpath_T, meta_csv_path).get_data()
        _, _, test_dl_ft_dm = scada_ds_norm_ft_dm["torch_dataloaders"]


    ##################################
    #       LOAD NBMs & MODELS       #
    ##################################

    # 1) TARGET DOMAIN: NBM trained on *full dataset* (its scores will be our ground truth)
    model_save_path = pathlib.Path.cwd().joinpath("saves", "NBM", WT_NAME_T, f"AE_model_{WT_NAME_T}.pt")
    target_nbm_representative = load_pretrained_NBM(model_save_path, model_in_ch = len(config_base["x_features"]), device=device)

    # load corresponding threshold
    with open(pathlib.Path.cwd().joinpath("saves", "NBM", WT_NAME_T, "normal_data_results.json")) as json_file: results = json.load(json_file)
    target_nbm_representative_threshold = results["val_N"]["mae"]["q3"] + ( 3 *  (results["val_N"]["mae"]["q3"] - results["val_N"]["mae"]["q1"]))


    # 2) TARGET DOMAIN: NBM trained on *scarce*, partial data
    if not args.NO_SCARCE:
        model_save_path = pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_T[2:], f"AE_model_{WT_NAME_T}.pt")
        target_nbm_scarce = load_pretrained_NBM(model_save_path, model_in_ch = len(config_base["x_features"]), device=device)

        # corresponding NBM-specific threshold
        with open(pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_T[2:], "normal_data_results.json")) as json_file: results = json.load(json_file)
        target_nbm_scarce_threshold = results["val_N"]["mae"]["q3"] + ( 3 *  (results["val_N"]["mae"]["q3"] - results["val_N"]["mae"]["q1"]))


    # 3) TARGET DOMAIN: source NBM fine-tuned on *scarce* target data
    if not args.NO_FINETUNE:
        model_name = f"AE_model_S_{WT_NAME_S}_FT_T_{WT_NAME_T}_{args.SCARCITY}.pt"
        model_save_path = pathlib.Path.cwd().joinpath("saves", "finetune", save_dir_S, save_dir_T, model_name)
        target_nbm_finetuned = load_pretrained_NBM(model_save_path, model_in_ch = len(config_base["x_features"]), device=device)
        
        # threshold
        with open(pathlib.Path.cwd().joinpath("saves", "finetune", save_dir_S, save_dir_T, "normal_data_results.json")) as json_file: results = json.load(json_file)
        target_nbm_finetuned_threshold = results["val_N"]["mae"]["q3"] + ( 3 *  (results["val_N"]["mae"]["q3"] - results["val_N"]["mae"]["q1"]))


    # 4) DOMAIN MAPPING NETWORK:
    # Relevant to us are the target->source mapped (genTS) and the source WT's NBM to evaluate the mapped data
    if not args.NO_DOMAINMAPPING:
        # i) source WT's NBM
        ae_model_S_name = f"AE_model_{WT_NAME_S}.pt"
        model_save_S_path = pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_S[2:], ae_model_S_name)
        source_nbm_representative = load_pretrained_NBM(model_save_S_path, model_in_ch = len(config_base["x_features"]), device=device)

        # load corresponding threshold
        with open(pathlib.Path.cwd().joinpath("saves", "NBM", save_dir_S[2:], "normal_data_results.json")) as json_file: results = json.load(json_file)
        source_nbm_representative_threshold = results["val_N"]["mae"]["q3"] + ( 3 *  (results["val_N"]["mae"]["q3"] - results["val_N"]["mae"]["q1"]))

        # ii) genTS to map target domain data to the source domain
        # T->S MAPPER FROM THE CYCLEGAN MODEL
        model_save_path = pathlib.Path.cwd().joinpath("saves", "mapping", save_dir_S, save_dir_T, "genTS.pt")
        genTS = load_pretrained_mapper(model_save_path, device=device)




    ######################################
    #    MODEL EVALUATION AND COMPARISON #
    ######################################
    results = {}

    # anomaly scores (reconstruction errors) for each sample in the test set
    # for i) the target WT NBM trained on its full, representative training data
    scores_representative_nbm =  np.asarray(nbm_eval.get_reconstr_errors(target_nbm_representative, test_dl_representativeNBM, device)["mae"])
    # convert to binary threshold exceedance (1 (positive) if score >= threshold, 0 (negative) else)
    thresh_scores_representative_nbm = np.int8(scores_representative_nbm >= target_nbm_representative_threshold)

    # anomaly scores from the target WT NBM trained on scarce data
    if not args.NO_SCARCE:
        scores_scarce_nbm =  np.asarray(nbm_eval.get_reconstr_errors(target_nbm_scarce, test_dl_scarce, device)["mae"])
        thresh_scores_scarce = np.int8(scores_scarce_nbm >= target_nbm_scarce_threshold)
        
        # calculate the threshold score similarity between the scarce NBM & the representative NBM (ground truth)
        # (see nbm_eval)
        scarce_results = nbm_eval.threshold_similarity_performance(thresh_scores_representative_nbm, thresh_scores_scarce)
        results["scarce"] = scarce_results 

    #  anomaly scores from the fine-tuned NBM
    if not args.NO_FINETUNE:
        scores_finetuned_nbm =  np.asarray(nbm_eval.get_reconstr_errors(target_nbm_finetuned, test_dl_ft_dm, device)["mae"])
        thresh_scores_ft = np.int8(scores_finetuned_nbm >= target_nbm_finetuned_threshold)
        
        # threshold similarity
        ft_results = nbm_eval.threshold_similarity_performance(thresh_scores_representative_nbm, thresh_scores_ft)
        results["finetune"] = ft_results 

    # CYCLEGAN MODEL
    if not args.NO_DOMAINMAPPING:
        # NOTE: note the different evaluation function which first performs a mapping!
        scores_dm =  np.asarray(mapping_eval.map_and_reconstruct(genTS, source_nbm_representative, test_dl_ft_dm, device)["mae"])
        thresh_scores_dm = np.int8(scores_dm >= source_nbm_representative_threshold)

        # threshold similarity
        dm_results = nbm_eval.threshold_similarity_performance(thresh_scores_representative_nbm, thresh_scores_dm)
        results["domainmapping"] = dm_results



    # 3) PUT IT INTO A DATAFRAME
    results_df = pd.DataFrame.from_dict(results)
    # & save the df
    results_path = pathlib.Path.cwd().joinpath("results", save_dir_S, f"T_{SITE_NAME_T}_WT_{WT_ID_T}")
    results_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path.joinpath(f"{save_dir_S}_{save_dir_T}_results.csv"))
    
    print(results_df)

    # clean up
    if not args.NO_DOMAINMAPPING: del genTS 
    if not args.NO_SCARCE: del target_nbm_scarce
    if not args.NO_FINETUNE: del target_nbm_finetuned
    if device !="cpu": torch.cuda.empty_cache()
    print("\n\n\n-----FINISHED---------")

if __name__ == "__main__":
    main(args)


