import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import math
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

class EndovisDataset(Dataset):
    def __init__(self, path_dict, need_aug=True, frame_len=1):
        """
        :param pathDict: 必须包括"img", "gt"字段, 以及对应的数据集列表
        """
        super().__init__()

        assert len(path_dict["img"]) == len(path_dict["gt"])

        self.path_dict = path_dict
        self.data_aug = need_aug
        self.frame_len = frame_len
        self.model_size = (224, 224)    # limitted by VGG16 backbone

        # 预先计算dataset个数和包含的图片的个数
        assert len(path_dict["img"]) > 0
        self.dataset_len = len(path_dict["img"])
        dum_dataset_path = path_dict["img"][-1]
        self.images_count = len(os.listdir(dum_dataset_path))

        self.__make_dataset()

    def augment_transform(self, images):
        return images


    def general_transform(self, img):
        """
        :param img: PIL image
        :return:
        """
        image = TF.crop(img, 36, 328, 1010, 1264)   # (1264, 1010)
        image = TF.resize(image, self.model_size)
        return image

    def __make_dataset(self):
        datadict = self.path_dict
        assert "img" in datadict and "gt" in datadict

        # 按照dataset目录加载图片的绝对路径, 后续可以设置具体需要加载的dataset
        datasets = datadict["img"]
        datasets.sort()

        images = []

        for dataset in datasets:
            img_path = dataset
            imgs = os.listdir(img_path)
            imgs.sort()  # 保证按时间顺序加载帧
            temp_images = []
            for img in imgs:
                img_route = os.path.join(img_path, img)
                temp_images.append(img_route)
            images.append(temp_images)

        # 按照dataset目录加载ground truth
        datasets = datadict["gt"]
        datasets.sort()

        ground_truths = []

        for dataset in datasets:
            img_path = dataset
            imgs = os.listdir(img_path)
            imgs.sort()  # 保证按时间顺序加载帧
            temp_ground_truths = []
            for img in imgs:
                img_route = os.path.join(img_path, img)
                temp_ground_truths.append(img_route)
            ground_truths.append(temp_ground_truths)

        self.images = images
        self.ground_truths = ground_truths

    def __len__(self):
        # 数据集大小应当等于：dataset_count*(ceil(image_count/frame_len))
        length = self.dataset_len * (self.images_count - self.frame_len + 1)
        return length

    def __getitem__(self, index):
        # pick out指定位置的图片路径，再进行真正的图片加载
        frame_count = self.images_count - self.frame_len + 1
        dataset_index = index // frame_count
        start = index % frame_count
        print("{} => dataset {} and offset {}".format(index, dataset_index, start))
        images = []
        ground_truths = []
        filenames = []

        # 获取本次图片所属的dataset，不允许跨数据集获取
        dataset_dir = os.path.dirname(self.images[dataset_index][0])
        datasets = [os.path.basename(dataset_dir)]

        # 真正加载图片
        for i in range(self.frame_len):
            raw_image = self.general_transform(Image.open(self.images[dataset_index][start+i]))    # 裁黑边并进行resize
            raw_ground_truth = self.general_transform(Image.open(self.ground_truths[dataset_index][start+i]))  # ground truth转为灰度图用于最终loss的计算

            print("loading image and ground truth from {}...".format(datasets[0]))
            print(self.images[dataset_index][start+i])
            print(self.ground_truths[dataset_index][start+i])

            filenames.append(os.path.basename(self.images[dataset_index][start+i]).split(".")[0])

            # data augmentation
            if self.data_aug:
                raw_image = self.augment_transform(raw_image)
                raw_ground_truth = self.augment_transform(raw_ground_truth)

            # 最终转化为0-1之间的数值的tensor，必须是pil image或者可以被认为是图片shape的numpy。
            image_tensor = TF.to_tensor(raw_image)
            ground_truth_tensor = TF.to_tensor(raw_ground_truth)

            images.append(image_tensor)
            ground_truths.append(ground_truth_tensor)

        images = torch.stack(images, 0)     # 增加batch size维度，共计四维tensor。
        ground_truths = torch.stack(ground_truths, 0)

        return images, ground_truths, filenames, datasets









