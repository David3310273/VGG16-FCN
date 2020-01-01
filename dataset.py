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
    def __init__(self, path_dict, need_aug=True, frame_len=1, training_ratio=1):
        """
        :param pathDict: 必须包括"img", "gt"字段
        """
        super().__init__()

        assert 0 < training_ratio <= 1
        assert len(os.listdir(path_dict["img"])) == len(os.listdir(path_dict["gt"]))

        self.path_dict = path_dict
        self.data_aug = need_aug
        self.frame_len = frame_len
        self.training_ratio = training_ratio
        self.model_size = (224, 224)

        # 预先计算dataset个数和包含的图片的个数
        assert len(os.listdir(path_dict["img"])) > 0
        self.dataset_len = len(os.listdir(path_dict["img"]))
        dum_dataset_path = os.path.join(path_dict["img"], os.listdir(path_dict["img"])[-1])
        self.images_count = len(os.listdir(dum_dataset_path))

        self.__make_dataset()

    def augment_transform(self, img):
        return img

    def general_transform(self, img):
        """
        :param img: PIL image
        :return:
        """
        image = TF.crop(img, 36, 328, 1010, 1264)   # (1264, 1010)
        image = TF.resize(image, self.model_size)
        return image

    def _get_dataset(self, dataset):
        length = len(dataset)
        training_length = math.floor(length*self.training_ratio)
        return dataset[:training_length]

    def __make_dataset(self):
        datadict = self.path_dict
        assert "img" in datadict and "gt" in datadict

        # 按照dataset目录加载图片的绝对路径, 后续可以设置具体需要加载的dataset
        datasets = os.listdir(datadict["img"])
        datasets = self._get_dataset(datasets)
        datasets.sort()

        images = []

        for dataset in datasets:
            img_path = os.path.join(datadict["img"], dataset)
            imgs = os.listdir(img_path)
            imgs.sort()  # 保证按时间顺序加载帧
            for img in imgs:
                img_path = os.path.join(img_path, img)
                images.append(img_path)

        # 按照dataset目录加载ground truth
        datasets = os.listdir(datadict["gt"])
        datasets = self._get_dataset(datasets)
        datasets.sort()

        ground_truths = []

        for dataset in datasets:
            img_path = os.path.join(datadict["gt"], dataset)
            imgs = os.listdir(img_path)
            imgs.sort()  # 保证按时间顺序加载帧
            for img in imgs:
                img_path = os.path.join(img_path, img)
                ground_truths.append(img_path)

        self.images = images
        self.ground_truths = ground_truths

    def __len__(self):
        # 数据集大小应当等于：dataset_count*(ceil(image_count/frame_len))
        length = self.dataset_len * (math.ceil(self.images_count / self.frame_len))
        return length

    def __getitem__(self, index):
        # pick out指定位置的图片，再进行真正的图片加载
        frame_count = math.ceil(self.images_count / self.frame_len)
        dataset_index = index // frame_count
        start = dataset_index*frame_count

        images = []
        ground_truths = []

        # 真正加载图片
        for i in range(self.frame_len):
            raw_image = self.general_transform(Image.open(self.images[start+i])) # 裁黑边并进行resize
            raw_ground_truth = self.general_transform(Image.open(self.ground_truths[start+i]).convert("L")) # ground truth转为灰度图用于最终loss的计算

            print("loading image...")
            print(self.images[start+i])
            print("loading ground truths...")
            print(self.ground_truths[start+i])

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

        return images, ground_truths









