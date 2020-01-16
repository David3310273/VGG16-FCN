import torch
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from PIL import Image
import configparser
import os
import time
from processing import *

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))
root = config["app"]["debug_output"]
epoch_output_root = config["training"]["epoch_output"]


def visualize_middle_map(x, filename):
    # image_tensors = x.permute(0, 2, 3, 1)
    if not os.path.exists(root):
        os.mkdir(root)
    for image_tensor in x:
        target = os.path.join(root, filename)
        image = TF.to_pil_image(image_tensor)
        image.save(target)


def write_training_images(data_loader, model, device, iter_num):
    """
    写入每一个epoch中的训练图片，neg和pos都进行存储
    目录结构：/root/pos_gt/iter_num/datasets/...
    :param image: 模型输出的tensor
    :param epoch: 轮数
    :return:
    """
    with torch.no_grad():
        model.eval()
        for idx, data in enumerate(data_loader):
            images = torch.squeeze(data[0], 1).to(device)
            filenames = data[4]  # might be multiple frames here, so it's two dimension array here.
            dataset = data[5][0][0]  # get the dataset name
            assert images.shape[1:] == (3, 224, 224)  # format: (batch_size, frame_len, c, h, w)
            outputs = model(images)

            if not os.path.exists(epoch_output_root):
                os.mkdir(epoch_output_root)
            if not os.path.exists(os.path.join(epoch_output_root, "pos_gt")):
                os.mkdir(os.path.join(epoch_output_root, "pos_gt"))
            if not os.path.exists(os.path.join(epoch_output_root, "neg_gt")):
                os.mkdir(os.path.join(epoch_output_root, "neg_gt"))

            pos_dataset_root = os.path.join(epoch_output_root, "pos_gt")
            neg_dataset_root = os.path.join(epoch_output_root, "neg_gt")

            if not os.path.exists(os.path.join(pos_dataset_root, iter_num)):
                os.mkdir(os.path.join(pos_dataset_root, iter_num))
            if not os.path.exists(os.path.join(neg_dataset_root, iter_num)):
                os.mkdir(os.path.join(neg_dataset_root, iter_num))

            pos_root = os.path.join(pos_dataset_root, iter_num)
            neg_root = os.path.join(neg_dataset_root, iter_num)

            for key, image in enumerate(outputs):
                filename = filenames[0][0]
                image = (F.sigmoid(image)).detach().cpu()
                print("writing image {} in dataset {} in iteration {}...".format(filename, dataset, iter_num))
                if not os.path.exists(os.path.join(pos_root, dataset)):
                    os.mkdir(os.path.join(pos_root, dataset))
                if not os.path.exists(os.path.join(neg_root, dataset)):
                    os.mkdir(os.path.join(neg_root, dataset))
                pos_img_root = os.path.join(pos_root, dataset, "{}.png".format(filename))
                neg_img_root = os.path.join(neg_root, dataset, "{}.png".format(filename))
                output = 255*binarify(image)
                neg_output = 255*binarify(1-image)
                output = TF.to_pil_image(output).convert("L")
                neg_output = TF.to_pil_image(neg_output).convert("L")
                output.save(pos_img_root)
                neg_output.save(neg_img_root)


def visualize_outlier(outlier_root, img, output, mask, fake_gt, epoch, dataset, filename):
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
    output = TF.to_pil_image(output)
    mask = TF.to_pil_image(mask)
    fake_gt = TF.to_pil_image(fake_gt)
    img_path = os.path.join(outlier_root, str(epoch), dataset,  "img_{}.png".format(filename[0]))
    output_path = os.path.join(outlier_root, str(epoch), dataset, "output_{}.png".format(filename[0]))
    mask_path = os.path.join(outlier_root, str(epoch), dataset, "mask_{}.png".format(filename[0]))
    fake_gt_path = os.path.join(outlier_root, str(epoch), dataset, "fake_gt_{}.png".format(filename[0]))
    img.save(img_path)
    output.save(output_path)
    mask.save(mask_path)
    fake_gt.save(fake_gt_path)



