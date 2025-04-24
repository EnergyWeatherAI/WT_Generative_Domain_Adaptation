from torch import save, load
from models import base_models, domain_mapping_models


def save_checkpoint(path, model, optimizer, epoch, tr_loss):
    '''
    Saves a model state with state dictionary and additional info to the specified path.
    '''
    save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'loss': tr_loss,
            }, path)



def load_pretrained_NBM(model_save_path, model_in_ch, device):
    '''
    Initializes a new ae-NBM and loads the weights from the specified path.
    Returns the model in eval mode and transferred to the device.
    '''
    ae_model = base_models.base_AE_CNN(in_channels=model_in_ch)
    model_checkpoint = load(model_save_path)
    ae_model.load_state_dict(model_checkpoint["model_state_dict"])
    ae_model = ae_model.to(device)
    ae_model = ae_model.eval();
    return ae_model

def load_pretrained_mapper(model_save_path, device):
    '''
    Initializes a new generator (mapper) and loads the weights from the specified path.
    Returns the model in eval mode and transferred to the device.
    '''
    gen = domain_mapping_models.GeneratorTCN()
    model_checkpoint = load(model_save_path)
    gen.load_state_dict(model_checkpoint["model_state_dict"])
    gen = gen.to(device)
    gen = gen.eval();
    return gen