import numpy as np 
import torch

###############################
# *helper* functions to evaluate a trained autoencoder-based normal behavior model.
# The actual evaluation is part of the evaluate_models.py script, please refer there for further general evaluation documentation.
###############################


def get_reconstr_errors(model, dl, device):
    '''
    Given a trained torch autoencoder-NBM model with batches of data from a dataloader, it returns a dictionary containing the reconstruction error for each sample. 
    
    Args:
        model (torch model): A trained torch autoencoder-NBM model.
        dl (dataloader): Dataloader with batches of data to calculate the reconstruction error
        device (torch device): Specify cuda device or cpu to perform calculations on.  
    '''

    # prep
    model.to(device).eval()
    error_dict = {
        "mse": [],
        "mae": []
    }

    # for each batch of data from the dataloader, 
    # i ) obtain the reconstructions from the autoencoder model
    # ii) Calculate the mse and mae between the original and reconstructions
    # iii) Store all reconstruction errors in a dictionary 
    with torch.no_grad():
        for x in iter(dl):
            x = x.to(device)
            reconstr = model(x) 

            mae_err = np.array(torch.mean(torch.abs(x - reconstr), dim=[1,2]).cpu()) # reduce datapoint-wise error to mean error per sample
            error_dict["mae"].extend(list(mae_err))
            mse_err = np.array(torch.mean(torch.square(x - reconstr), dim=[1,2]).cpu())
            error_dict["mse"].extend(list(mse_err))

    return error_dict


def get_result_statistics(model, dl, device):
    ''' 
    Summarizes results from get_reconstr_errors by additionally calculating the averaged mean, std, q1, and q3. (equivalent args)
    Used to determine the model-specific threshold after training a model. 
    '''
    error_dict = get_reconstr_errors(model, dl, device)

    mae_errors = np.asarray(error_dict["mae"])
    mse_errors = np.asarray(error_dict["mse"])

    mae_mean, mse_mean = np.mean(mae_errors), np.mean(mse_errors)
    mae_std, mse_std = np.std(mae_errors), np.std(mse_errors)
    mae_q3, mse_q3 = np.quantile(mae_errors, 0.75), np.quantile(mse_errors, 0.75)
    mae_q1, mse_q1 = np.quantile(mae_errors, 0.25), np.quantile(mse_errors, 0.25)

    results = {"mae": {}, "mse": {}}
    results["mae"] = {"mean": float(mae_mean), "std": float(mae_std), "q3": float(mae_q3), "q1": float(mae_q1)}
    results["mse"] = {"mean": float(mse_mean), "std": float(mse_std), "q3": float(mse_q3), "q1": float(mse_q1)}
    return results


def threshold_similarity_performance(y_actual, y_hat):
    '''
    Calculates and returns threshold similarity statistics (incl. F1-score) between two binary threshold-exceedance timeseries (e.g., from two different NBMs).
       
    Args:
        y_actual (numpy array): Ground truth (non-scarce NBM) of the threshold-exceedance: np.arr([0, 1, 0, ...]). An array showing whether for each sample the anomaly score (reconstruction error) was above the model-specific threshold (1) or below (0)
        y_hat (numpy array): Corresponding scores of the evaluated model (e.g., scarce NBM) of the threshold-exceedance: np.arr([0, 1, 0, ...]).

    The threshold similarity is calculated based on the correspondence of anomaly scores in terms of the threshold of the models. 
    
    Example:
        Ground truth NBM has the following anomaly scores: [0.3, 0.4, 0.5] with a threshold of 0.35.
        In that case, y_actual is the following array: [0 (negative), 1 (positive), 1 (positive)]

        The to-be-compared model has anomaly scores [0.5, 0.3, 0.7] with its specific threshold of 0.49
        In that case, the provided y_hat is the following array: [1, 0, 1]

        This function would then calculate the similarity according to various measures between [0,1,1] and [1,0,1]. 
    '''

    TP, FP, TN, FN = 0, 0, 0, 0
    P = int((y_actual==1).sum())
    N = int((y_actual==0).sum())

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    if TN == 0:
        return {"TP": -1, "FP": -1, "TN": -1, "FN": -1, "P": -1, "N": -1, "TPR": -1, "FPR": -1, "FNR": -1, "TNR": -1,
                        "ACC": -1, "PPV": -1, "NPV": -1, "F1": -1}


    
    TPR = TP/P if P > 0 else 1.0 
    FPR = FP/N 
    FNR = FN/P if P > 0 else 1.0
    TNR = TN/N 

    ACC = (TP+TN)/(P+N)
    PPV = TP/(TP+FP) if (TP+FP) > 0 else 1.0
    NPV = TN/(TN+FN)

    F1 = ((2 * TP) / ((2 * TP) + FN + FP))

    SENS = TP / (TP + FN) if (TP+FN) > 0 else -1.0

    return {"TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
                     "P": int(P), "N": int(N),
                        "TPR": TPR*100, "FPR": FPR*100, "FNR": FNR*100, "TNR": TNR*100,
                        "ACC": ACC*100, "PPV": PPV*100, "NPV": NPV*100,
                        "F1": F1*100, "SENS": SENS*100}



