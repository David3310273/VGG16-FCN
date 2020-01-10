import torch
from torchvision.transforms import functional as TF
from PIL import Image
import configparser
import os
import time
from processing import *

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))
root = config["app"]["debug_output"]
outlier_root = config["app"]["outlier_dir"]
epoch_output_root = config["training"]["epoch_output"]


def visualize_middle_map(x, filename):
    # image_tensors = x.permute(0, 2, 3, 1)
    if not os.path.exists(root):
        os.mkdir(root)
    for image_tensor in x:
        target = os.path.join(root, filename)
        image = TF.to_pil_image(image_tensor)
        image.save(target)


def write_training_images(image, epoch, dataset, filenames):
    """
    写入每一个epoch中的训练图片，neg和pos都进行存储
    :param image: 模型输出的tensor
    :param epoch: 轮数
    :return:
    """
    if not os.path.exists(epoch_output_root):
        os.mkdir(epoch_output_root)
    if not os.path.exists(os.path.join(epoch_output_root, dataset)):
        os.mkdir(os.path.join(epoch_output_root, dataset))
    if not os.path.exists(os.path.join(epoch_output_root, dataset)):
        os.mkdir(os.path.join(epoch_output_root, dataset))

    pos_dataset_root = os.path.join(epoch_output_root, dataset)
    neg_dataset_root = os.path.join(epoch_output_root, dataset)

    if not os.path.exists(os.path.join(pos_dataset_root, "pos")):
        os.mkdir(os.path.join(pos_dataset_root, "pos"))
    if not os.path.exists(os.path.join(neg_dataset_root, "neg")):
        os.mkdir(os.path.join(neg_dataset_root, "neg"))

    pos_root = os.path.join(pos_dataset_root, "pos")
    neg_root = os.path.join(pos_dataset_root, "neg")

    for filename in filenames:
        if not os.path.exists(os.path.join(pos_root, str(epoch))):
            os.mkdir(os.path.join(pos_root, str(epoch)))
        if not os.path.exists(os.path.join(neg_root, str(epoch))):
            os.mkdir(os.path.join(neg_root, str(epoch)))
        pos_img_root = os.path.join(pos_root, str(epoch), "{}.png".format(filename))
        neg_img_root = os.path.join(neg_root, str(epoch), "{}.png".format(filename))
        output = 255*binarify(image)
        neg_output = 255*binarify(1-image)
        output = TF.to_pil_image(output).convert("L")
        neg_output = TF.to_pil_image(neg_output).convert("L")
        output.save(pos_img_root)
        neg_output.save(neg_img_root)


def visualize_outlier(img, output, mask, fake_gt, epoch, dataset, filename):
    """
    记录性能异常的输入输出，并标注epoch
    :param img: 输入
    :param output: 输出
    :param fake_gt:
    :param epoch: 迭代轮数
    :return:
    """
    if not os.path.exists(outlier_root):
        os.mkdir(outlier_root)

    if not os.path.exists(os.path.join(outlier_root, str(epoch))):
        os.mkdir(os.path.join(outlier_root, str(epoch)))

    if not os.path.exists(os.path.join(outlier_root, str(epoch), dataset)):
        os.mkdir(os.path.join(outlier_root, str(epoch), dataset))

    img = TF.to_pil_image(img)
    output = 255*binarify(output)
    output = TF.to_pil_image(output).convert("L")
    mask = TF.to_pil_image(mask).convert("L")
    fake_gt = 255*binarify(fake_gt)
    fake_gt = TF.to_pil_image(fake_gt).convert("L")
    img_path = os.path.join(outlier_root, str(epoch), dataset,  "img_{}.png".format(filename[0]))
    output_path = os.path.join(outlier_root, str(epoch), dataset, "output_{}.png".format(filename[0]))
    mask_path = os.path.join(outlier_root, str(epoch), dataset, "mask_{}.png".format(filename[0]))
    fake_gt_path = os.path.join(outlier_root, str(epoch), dataset, "fake_gt_{}.png".format(filename[0]))
    img.save(img_path)
    output.save(output_path)
    mask.save(mask_path)
    fake_gt.save(fake_gt_path)



