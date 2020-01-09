import torch
import random

# 根据阈值，将Tensor二值化为0-1map
def binarify(output, threshold=0.5):
    ones_temp = torch.ones_like(output, dtype=torch.uint8)
    zeros_temp = torch.zeros_like(output, dtype=torch.uint8)

    final_output = torch.where(output > threshold, ones_temp, zeros_temp)
    return final_output

# 给定多个list，以相同的方式就地shuffle多个数组
def shuffle_lists(*lists):
    seed = random.random()
    for array in lists:
        random.seed(seed)
        random.shuffle(array)


