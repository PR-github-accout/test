import os
import math
import argparse
import numpy as np
import tensorboard

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import torchsummary
from thop import profile
from fvcore.nn import FlopCountAnalysis

from my_dataset import train_Datasets, val_Datasets
from model import Model
from pointnet import PointNetCls as pointnet
from torch.utils.data import DataLoader
from utils import train_one_epoch, evaluate

def main(epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    data_transform = {
        "train": transforms.Compose([#transforms.RandomHorizontalFlip(),
                                     #transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])]),
        "val": transforms.Compose([
                                   #transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5])])}

    # train_dataset = train_Datasets(r"E:\PycharmProjects\Transformer\swin_transformer1\Dataset\train\data",
    #                                transform=data_transform["train"])
    # val_dataset = val_Datasets(r"E:\PycharmProjects\Transformer\swin_transformer1\Dataset\val\data",
    #                            transform=data_transform["val"])

    train_dataset = train_Datasets(r"E:\PycharmProjects\Transformer\swin_transformer1\Dataset\train\data")
    val_dataset = val_Datasets(r"E:\PycharmProjects\Transformer\swin_transformer1\Dataset\val\data")

    train_loader = DataLoader(dataset=train_dataset, batch_size=30, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=30,shuffle=False)

    model = Model().to(device)
    # model = pointnet().to(device)
    # model.load_state_dict(torch.load("weights/model-639.pth", map_location=device))

    pg = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.Adam(pg, lr = 0.0001, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(epochs):
        # train
        train_loss, train_acc, train_loss1, train_acc1 = train_one_epoch(model=model,
                                                                         optimizer=optimizer,
                                                                         data_loader=train_loader,
                                                                         device=device,
                                                                         epoch=epoch)

        scheduler.step()


        train_Loss_list = []
        train_Loss_list1 = []

        train_Accuracy_list = []
        train_Accuracy_list1 = []


        train_Loss_list.append(train_loss)
        train_Loss_list1.append(train_loss1)

        train_Accuracy_list.append((100 * train_acc))
        train_Accuracy_list1.append((100 * train_acc1))


        Loss0 = np.array(train_Loss_list)
        Loss1 = np.array(train_Loss_list1)

        Acc0 = np.array(train_Accuracy_list)
        Acc1 = np.array(train_Accuracy_list1)

        np.save('history/loss_train8/epoch_{}'.format(epoch), Loss0)
        # np.save('history/loss_train3/epoch_{}'.format(epoch), Loss1)

        np.save('history/acc_train8/epoch_{}'.format(epoch), Acc0)
        # np.save('history/acc_train3/epoch_{}'.format(epoch), Acc1)


        # validate
        val_loss, val_acc, val_loss1, val_acc1 = evaluate(model=model,
                                                          data_loader=val_loader,
                                                          device=device,
                                                          epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

        val_Loss_list = []
        val_Loss_list1 = []

        val_Accuracy_list = []
        val_Accuracy_list1 = []


        val_Loss_list.append((val_loss))
        val_Loss_list1.append((val_loss1))

        val_Accuracy_list.append((100 * val_acc))
        val_Accuracy_list1.append((100 * val_acc1))

        Loss0 = np.array(val_Loss_list)
        Loss1 = np.array(val_Loss_list1)

        Acc0 = np.array(val_Accuracy_list)
        Acc1 = np.array(val_Accuracy_list1)

        np.save('history/loss_val8/epoch_{}'.format(epoch), Loss0)
        # np.save('history/loss_val3/epoch_{}'.format(epoch), Loss1)

        np.save('history/acc_val8/epoch_{}'.format(epoch), Acc0)
        # np.save('history/acc_val3/epoch_{}'.format(epoch), Acc1)



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    torchsummary.summary(model, (1, 2, 650))
    main(100)

       # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       # model = create_model_base(num_classes=650, has_logits=False).to(device)
       # t = torch.randn(1, 1, 650, 2).to(device)
       # flops1 = FlopCountAnalysis(model, t)
       # print("Self-Attention FLOPs:", flops1.total())
       #torchsummary.summary(model, (1,650,2))
