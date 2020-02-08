import os
import random
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import math
import torchvision.transforms.functional as TF
from processing import augment_transform, threshold_by_ostu


class IsicDataset(Dataset):
    def __init__(self, path_dict, need_aug=True, is_test=False):
        """
        :param pathDict: 必须包括"train_img", "train_gt"字段, 以及对应的数据集列表
        """
        super().__init__()

        self.path_dict = path_dict
        self.data_aug = need_aug
        self.model_size = (224, 224)    # limitted by VGG16 backbone
        self.is_test = is_test

        # 预先计算dataset个数和包含的图片的个数
        dum_dataset_path = path_dict["train_img"] if not is_test else path_dict["test_img"]
        self.images_count = len(os.listdir(dum_dataset_path))

        self.__make_dataset()

    def general_transform(self, img):
        """
        :param img: PIL image
        :return:
        """
        image = TF.resize(img, self.model_size)
        return image

    def __make_dataset(self):
        datadict = self.path_dict
        assert "train_img" in datadict and "train_gt" in datadict
        assert "train_pos_gt" in datadict
        assert "test_img" in datadict and "test_gt" in datadict
        assert "test_pos_gt" in datadict

        # 按照dataset目录加载图片的绝对路径, 后续可以设置具体需要加载的dataset
        img_path = datadict["train_img"]
        images = []

        imgs = os.listdir(img_path)
        imgs.sort()  # 保证按时间顺序加载帧
        for img in imgs:
            img_route = os.path.join(img_path, img)
            images.append(img_route)

        # 按照dataset目录加载ground truth
        img_path = datadict["train_gt"]

        ground_truths = []

        imgs = os.listdir(img_path)
        imgs.sort()  # 保证按时间顺序加载帧
        for img in imgs:
            img_route = os.path.join(img_path, img)
            ground_truths.append(img_route)

        img_path = datadict["train_pos_gt"]
        pos_gts = []

        if img_path:
            imgs = os.listdir(img_path)
            imgs.sort()  # 保证按时间顺序加载帧
            for img in imgs:
                img_route = os.path.join(img_path, img)
                pos_gts.append(img_route)

        # 加载测试数据集

        img_path = datadict["test_img"]
        test_images = []

        imgs = os.listdir(img_path)
        imgs.sort()  # 保证按时间顺序加载帧
        for img in imgs:
            img_route = os.path.join(img_path, img)
            test_images.append(img_route)

        # 按照dataset目录加载test ground truth
        img_path = datadict["test_gt"]

        test_ground_truths = []

        imgs = os.listdir(img_path)
        imgs.sort()  # 保证按时间顺序加载帧
        for img in imgs:
            img_route = os.path.join(img_path, img)
            test_ground_truths.append(img_route)

        img_path = datadict["test_pos_gt"]
        test_pos_gts = []

        if img_path:
            imgs = os.listdir(img_path)
            imgs.sort()  # 保证按时间顺序加载帧
            for img in imgs:
                img_route = os.path.join(img_path, img)
                test_pos_gts.append(img_route)

        self.train_images = images
        self.train_ground_truths = ground_truths
        self.train_pos_gts = pos_gts

        self.test_images = test_images
        self.test_ground_truths = test_ground_truths
        self.test_pos_gts = test_pos_gts


    def __len__(self):
        return len(self.train_images) if not self.is_test else len(self.test_images)


    def threshold_transform(self, img):
        """
        传入0-255的灰度PIL图，给出分割后的PIL图片
        :param img:
        :return:
        """
        image = np.array(img)
        threshold = threshold_by_ostu(image)
        image[image > threshold] = 255
        image[image <= threshold] = 0

        return Image.fromarray(image)

    def __getitem__(self, index):
        # pick out指定位置的图片路径，再进行真正的图片加载
        print("{} => dataset {} and offset {}".format(index, "training" if not self.is_test else "testing", index))
        image = []
        ground_truth = []
        pos_gt = []
        filenames = []

        # 真正加载图片
        if not self.is_test:
            raw_image = self.general_transform(Image.open(self.train_images[index]))    # 裁黑边并进行resize
            raw_image = raw_image.filter(ImageFilter.MaxFilter(7))  # 对原图进行滤波处理
            raw_ground_truth = self.general_transform(Image.open(self.train_ground_truths[index]))  # ground truth转为灰度图用于最终loss的计算
            raw_pos_fake_ground_truth = self.general_transform(Image.open(self.train_pos_gts[index]))
            filenames = (self.train_images[index]).split(".")[0]
            raw_pos_fake_ground_truth = self.threshold_transform(raw_pos_fake_ground_truth)

            print("loading image and ground truth from...")
            print(self.train_images[index])
            print(self.train_ground_truths[index])
            print(self.train_pos_gts[index])
            print(filenames)
        else:
            raw_image = self.general_transform(Image.open(self.test_images[index]))  # 裁黑边并进行resize
            raw_image = raw_image.filter(ImageFilter.MaxFilter(7))  # 对原图进行滤波处理
            raw_ground_truth = self.general_transform(
                Image.open(self.test_ground_truths[index]))  # ground truth转为灰度图用于最终loss的计算
            raw_pos_fake_ground_truth = self.general_transform(Image.open(self.test_pos_gts[index]))
            filenames = (self.test_images[index]).split(".")[0]
            # 对fake ground truth做二值化以备训练
            raw_pos_fake_ground_truth = self.threshold_transform(raw_pos_fake_ground_truth)

            print("loading image and ground truth from...")
            print(self.test_images[index])
            print(self.test_ground_truths[index])
            print(self.test_pos_gts[index])
            print(filenames)

        # data augmentation
        if self.data_aug:
            raw_image, raw_ground_truth, raw_pos_fake_ground_truth = augment_transform(raw_image, raw_ground_truth, raw_pos_fake_ground_truth, target_size=self.model_size)

        # 最终转化为0-1之间的数值的tensor，必须是pil image或者可以被认为是图片shape的numpy。
        image = TF.to_tensor(raw_image)
        ground_truth = TF.to_tensor(raw_ground_truth)
        pos_gt = TF.to_tensor(raw_pos_fake_ground_truth)
        filename = (os.path.basename(filenames)).split(".")[0]

        return image, ground_truth, pos_gt, filename










