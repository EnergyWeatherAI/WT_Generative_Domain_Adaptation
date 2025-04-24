import numpy as np 
import torch

###############################
# *helper* functions to evaluate a trained domain-mapping model.
# The actual evaluation is part of the evaluate_models.py script, please refer there for further general evaluation documentation.
###############################

def map_and_reconstruct(mapper, autoencoder, dl, device):
    '''
    First maps batches of data to the other domain, then evaluates its reconstruction errors on that domain's NBM. Returns a dictionary with errors for each sample.
    
    Args:
        mapper (torch model): A generator/mapper of the domain mapping network to map in the direction of domain(provided batches) to the other domain. (Example: target domain)
        autoencoder (torch model): The trained NBM of the *other* domain, i.e., NOT from the domain of the batches. (Example: source domain NBM)
        dl (torch dataloader): Dataloader with batches from the domain to map  (Example: target domain test dataloader)
        device (torch device): cuda device or cpu to perform the calculations on.
    '''

    # prep
    error_dict = {
        "mse": [],
        "mae": []
    }

    # For each batch in the provided dataloader:
    # i) *Map* the batch to the other domain (continuing example: from target to source)
    # ii) Get the reconstruction error of the mapped data on the other domain's NBM (c. example: evaluate mapped(target) on the source domain NBM)
    # iii) Store MAE & MSE reconstruction errors for each sample in a dictionary
    
    loader = iter(dl)
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            x_mapped = mapper(x)

            reconstr_mapped = autoencoder(x_mapped)
            mae_err = np.array(torch.mean(torch.abs(x_mapped - reconstr_mapped), dim=[1,2]).cpu()) # reduce point-wise errors to mean error per sample
            error_dict["mae"].extend(list(mae_err))
            mse_err = np.array(torch.mean(torch.square(x_mapped - reconstr_mapped), dim=[1,2]).cpu())
            error_dict["mse"].extend(list(mse_err))

    return error_dict