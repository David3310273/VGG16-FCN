import torch


def getIOU(output, ground_truth):
    """
    输入必须介于0-1之间的数，返回数值
    :param output:
    :param ground_truth:
    :return:
    """
    ones_temp = torch.ones_like(output, dtype=torch.uint8)
    zeros_temp = torch.zeros_like(output, dtype=torch.uint8)

    output_discreted = torch.where(output > 0.5, ones_temp, zeros_temp)
    ground_truth_discreted = torch.where(ground_truth > 0.5, ones_temp, zeros_temp)

    # 先化成数值再计算，不能以向量的形式计算，
    intersection = torch.sum(output_discreted & ground_truth_discreted).item()
    union = torch.sum(output_discreted | ground_truth_discreted).item()

    return (intersection + 1e-15) / (union + 1e-15)
