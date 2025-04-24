import numpy as np 
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

####   
# Handles torch-related data processing. Converts SCADA numpy sequences into torch datasets with appropriate normalization functions and batched dataloaders.
####

class DatasetFromNumpy(Dataset):
    def __init__(self, x, transform=None):
        '''
        Returns a torch dataset object based on the SCADA sequences and the normalization function.

        Args:
            x: The extracted SCADA sequences as numpy arrays.
            transform: the torch transformation function to apply when retrieving an object from the dataset. Here: the normalization function.
        '''
        self.x = torch.from_numpy(x).type(torch.float32)
        self.transform = transform

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        ''' When retrieving a specific sequence from the torch dataset, apply the transformation (i.e., normalization) function'''
        x = self.x[idx]
        if self.transform: x = self.transform(x)
        return x


def get_torch_datasets(list_of_np_datasets, transform=None):
    # create a dataset for each numpy sequence dataset entry in the list, e.g. [tr_seqs, val_seqs, test_seqs] where the shape is typically tr_seqs = n_seq x ch x l 
    datasets = []
    for x in list_of_np_datasets: datasets.append(DatasetFromNumpy(x, transform))
    if len(list_of_np_datasets) == 1: return datasets[0]
    else: return datasets


def get_dataloaders(dataset_list, batch_sizes = [32, 32, 32], shuffles=[True, False, False], 
                                drop_last = [True, True, True]):
    # create a dataloader for each dataset in the list with the specified batch_size, shuffle, and drop_last flags.
    dataloaders = []
    for i, ds in enumerate(dataset_list): 
        dataloaders.append(DataLoader(ds, batch_size=batch_sizes[i], shuffle=shuffles[i], drop_last = drop_last[i], num_workers=4, pin_memory=True))

    if len(dataset_list) > 1: return dataloaders 
    else: return dataloaders[0]

def get_normalize_f(mins, maxs, a=-1, b=1):
    # create a torch-appropriate transformation function that applied min-max scaling in the range of [a, b]
    norm_f = lambda x: (b-a) * ((x-torch.broadcast_to(mins.unsqueeze(1), x.shape))/(torch.broadcast_to(maxs.unsqueeze(1), x.shape) - torch.broadcast_to(mins.unsqueeze(1), x.shape))) + a
    return norm_f

def get_unnormalize_f(mins, maxs, a=-1, b=1):
    # reverts an applied normalization to data | arguments must match the get_normalize_f args.
    unnorm_f = lambda x: (((x-a)/(b-a)) * (maxs.unsqueeze(1) - mins.unsqueeze(1))) + mins.unsqueeze(1)
    return unnorm_f



