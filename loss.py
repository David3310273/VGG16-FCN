import torch

def lossWithBCE(pos_weight=None):
    return torch.nn.BCEWithLogitsLoss(pos_weight)