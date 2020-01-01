from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF
import torch.optim as optim
import os
import configparser
from benchmarks import *

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))

epochs = int(config["training"]["epochs"])


def training(train_loader, test_loader, model, loss_fn):
    target = config["training"]["logdir"]
    writer = SummaryWriter(target)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if not os.path.exists(target):
        os.mkdir(target)

    if not os.path.exists("images"):
        os.mkdir("images")

    for i in range(epochs):
        total_loss = 0
        index = 0
        model.train()
        print("### the epoch {} start.... ###".format(i))
        for idx, data in enumerate(train_loader):
            images, ground_truths = torch.squeeze(data[0], 1), torch.squeeze(data[1], 1)
            assert images.shape[1:] == (3, 224, 224)    # format: (batch_size, frame_len, c, h, w)
            assert ground_truths.shape[1:] == (1, 224, 224)
            outputs = model(images)
            optimizer.zero_grad()
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
                assert images.shape[1:] == (3, 224, 224)  # format: (batch_size, frame_len, c, h, w)
                assert ground_truths.shape[1:] == (1, 224, 224)
                outputs = model(images)
                for key, output in enumerate(outputs):
                    # get iou at each epoch
                    iou += getIOU(output, ground_truths[key])
                    index += 1
                    # visualize the output
                    result_image = TF.to_pil_image(255*output).convert("L")     # (1, 224, 224)
                    result_image.save(os.path.join("images", "epoch_{}.png".format(i)))
            avg_iou = iou / index
            print("The iou at epoch {} is {}...".format(i, avg_iou))
            writer.add_scalar("train/iou", avg_iou, i)
    writer.close()



