from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF
from torch.nn import functional as F
import torch.optim as optim
import os
import configparser
from benchmarks import *
from visualization import *
import numpy as np

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))
epochs = int(config["training"]["epochs"])
is_debug = bool(config["app"]["debug"])


def testing(test_loader, model, loss_fn, epoch=0):
    target = config["testing"]["logdir"]
    writer = SummaryWriter(target)

    print("### Testing period....... ###")

    # 此处没有训练过程
    with torch.no_grad():
        model.eval()
        index = 0
        iou = 0
        for idx, data in enumerate(test_loader):
            images, ground_truths = torch.squeeze(data[0], 1), torch.squeeze(data[1], 1)
            assert images.shape[1:] == (3, 224, 224)  # format: (batch_size, frame_len, c, h, w)
            assert ground_truths.shape[1:] == (1, 224, 224)
            outputs = model(images)
            loss = loss_fn(outputs, ground_truths)
            print("The loss at epoch {} is {}...".format(epoch, loss))
            writer.add_scalar("test/bce_loss", loss, epoch)
            # calculate iou
            for key, output in enumerate(outputs):
                # get iou at each epoch
                iou += getIOU(output, ground_truths[key])
                index += 1
            avg_iou = iou / index
            print("The iou at epoch {} is {}...".format(epoch, avg_iou))
            writer.add_scalar("test/iou", avg_iou, epoch)
    writer.close()


def training(train_loader, test_loader, model, loss_fn):
    target = config["training"]["logdir"]
    writer = SummaryWriter(target)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if not os.path.exists(target):
        os.mkdir(target)

    for i in range(epochs):
        total_loss = 0
        index = 0
        model.train()
        print("### the epoch {} start.... ###".format(i))
        for idx, data in enumerate(train_loader):
            images, ground_truths = torch.squeeze(data[0], 1), torch.squeeze(data[1], 1)
            assert images.shape[1:] == (3, 224, 224)    # format: (batch_size, frame_len, c, h, w)
            assert ground_truths.shape[1:] == (1, 224, 224)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, ground_truths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            index += 1

        print("The loss of epoch {} is {}".format(i, total_loss/index))
        writer.add_scalar("train/bce_loss", total_loss/index, i)
        # output result image every epoch
        with torch.no_grad():
            model.eval()
            index = 0
            iou = 0
            for idx, data in enumerate(train_loader):
                images, ground_truths = torch.squeeze(data[0], 1), torch.squeeze(data[1], 1)
                filenames = data[2]     # might be multiple frames here, so it's two dimension array here.
                dataset = data[3][0][0] # get the dataset name
                assert images.shape[1:] == (3, 224, 224)  # format: (batch_size, frame_len, c, h, w)
                assert ground_truths.shape[1:] == (1, 224, 224)
                outputs = model(images)
                for key, output in enumerate(outputs):
                    # get iou at each epoch, using sigmoid to activate.
                    output_for_iou = F.sigmoid(output)
                    temp_iou = getIOU(output_for_iou, ground_truths[key])
                    # save the output after training for the next epoch
                    write_training_images(output_for_iou, i, dataset, filenames[key])
                    # record the outliers when iou is less than 0.5
                    if is_debug and temp_iou < 0.5:
                        visualize_outlier(images[key], output_for_iou, ground_truths[key], i, filenames[key])
                    iou += temp_iou
                    index += 1
            avg_iou = iou / index
            print("The iou at epoch {} is {}...".format(i, avg_iou))
            writer.add_scalar("train/iou", avg_iou, i)
        # 测试嵌套在每一个epoch训练完之后
        if not is_debug:
            testing(test_loader, model, loss_fn, i)
    writer.close()



