import torch
from torchvision.transforms import functional as TF
from PIL import Image
import configparser
import os
import time

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))
root = config["app"]["debug_output"]
outlier_root = config["app"]["outlier_dir"]


def visualize_middle_map(x, filename):
    # image_tensors = x.permute(0, 2, 3, 1)
    if not os.path.exists(root):
        os.mkdir(root)
    for image_tensor in x:
        target = os.path.join(root, filename)
        image = TF.to_pil_image(image_tensor)
        image.save(target)


def visualize_outlier(img, output, mask, epoch, filename):
    """
    记录性能异常的输入输出，并标注epoch
    :param img: 输入
    :param output: 输出
    :param epoch: 迭代轮数
    :return:
    """
    if not os.path.exists(outlier_root):
        os.mkdir(outlier_root)

    if not os.path.exists(os.path.join(outlier_root, str(epoch))):
        os.mkdir(os.path.join(outlier_root, str(epoch)))

    img = TF.to_pil_image(img)
    output = TF.to_pil_image(output).convert("L")
    mask = TF.to_pil_image(mask).convert("L")
    img_path = os.path.join(outlier_root, str(epoch), "img_{}.png".format(filename))
    output_path = os.path.join(outlier_root, str(epoch), "output_{}.png".format(filename))
    mask_path = os.path.join(outlier_root, str(epoch), "mask_{}.png".format(filename))
    img.save(img_path)
    output.save(output_path)
    mask.save(mask_path)



