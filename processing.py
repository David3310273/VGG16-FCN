import torch


def binarify(output, threshold=0.5):
    ones_temp = torch.ones_like(output, dtype=torch.uint8)
    zeros_temp = torch.zeros_like(output, dtype=torch.uint8)

    final_output = torch.where(output > threshold, ones_temp, zeros_temp)
    return final_output

