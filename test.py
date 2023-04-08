import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms

from my_dataset import test_Datasets
from model import Model as create_model
from pointnet import PointNetCls as pointnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = {
        "test": transforms.Compose([#transforms.RandomHorizontalFlip(),
                                     #transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])]),}

test_dataset = test_Datasets(r"E:\PycharmProjects\Transformer\swin_transformer1\Dataset\test\data")
test_loader = DataLoader(dataset=test_dataset, batch_size=300, shuffle=False)

model = create_model().to(device)
# model = pointnet().to(device)
model.load_state_dict(torch.load("weights/model.pth"))

def test():
    with torch.no_grad():
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        sample_num = 0
        for step, data in enumerate(test_loader):
            inputs, labels = data
            sample_num += inputs.shape[0]

            pred, pred1 = model(inputs.to(device))
            pred_classes = torch.max(pred, dim=1)[1]

            accu_num += ((pred_classes - labels.to(device)) ** 2 <= 1).sum()
            print('Accuracy on test set: %3f %%' %(100*accu_num/sample_num))
    #f = open(r'output.txt', 'w')

    #print(predicted, file=f)
    #f.close()

    #f1 = open(r'Probability.txt','w')
    #torch.set_printoptions(threshold=np.inf)
    #print(outputs.data,file=f1)
    #f1.close()
    #print(torch.max(outputs.data,dim=1))
    print(pred_classes)
    # pre,_ = torch.max(pred.data.cpu(),dim=1)
    # print(pre)
    plt.plot(pred_classes.data.cpu())
    plt.show()

if __name__ == '__main__':
    test()