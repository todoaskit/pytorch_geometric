import os
import sys
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet10  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import (NormalizeScale, LogCartesianAdj,
                                       RandomTranslate, CartesianAdj,
                                       RandomRotate)  # noqa
from torch_geometric.nn.modules import SplineConv, InvGraphConv  # noqa
from torch.nn import Linear as Lin # noqa
from torch_geometric.nn.functional import (sparse_voxel_max_pool,
                                           dense_voxel_max_pool)  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10')

transform = LogCartesianAdj(30)
#init_transform = Compose([RandomRotate(np.pi), NormalizeScale()])
init_transform = NormalizeScale()
train_dataset = ModelNet10(path, True, init_transform)
test_dataset = ModelNet10(path, False, init_transform)
batch_size = 6
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = InvGraphConv(1, 12, dim=3, kernel_size=5)
        self.conv2 = InvGraphConv(15, 20, dim=3, kernel_size=5)
        self.conv3 = InvGraphConv(23, 28, dim=3, kernel_size=5)
        self.conv4 = InvGraphConv(31, 36, dim=3, kernel_size=5)
        self.conv5 = InvGraphConv(39, 44, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(8 * 47, 10)

        self.att1 = Lin(32, 2)
        self.att2 = Lin(64, 2)
        self.att3 = Lin(64, 2)
        self.att4 = Lin(64, 2)

    def pool_args(self, mean, x):
        if not self.training:
            return 1 / mean, 0
        #size = [1 / random.uniform(mean - x, mean + x) for _ in range(3)]
        size = 1 / mean
        start = [random.uniform(-1 / (mean - x), 0) for _ in range(3)]
        #start = 0
        return size, start

    def forward(self, data):
        data.input, c_loss1 = self.conv1(data.adj, data.input)
        #att1 = self.att1(data.input)
        size, start = self.pool_args(32, 8)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input, c_loss2 = self.conv2(data.adj, data.input)

        #att2 = self.att2(data.input)
        size, start = self.pool_args(16, 4)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input, c_loss3 = self.conv3(data.adj, data.input)
        #att3 = self.att3(data.input)
        size, start = self.pool_args(8, 2)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input, c_loss4 = self.conv4(data.adj, data.input)
        #att4 = self.att4(data.input)
        size, start = self.pool_args(4, 1)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input, c_loss5 = self.conv5(data.adj, data.input)
        data, _ = dense_voxel_max_pool(data, 1, -0.5, 1.5)

        x = data.input.view(-1, self.fc1.weight.size(1))
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1), c_loss1 + c_loss2 + c_loss3 + \
               c_loss4 + c_loss5


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 2:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    if epoch == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 11:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    if epoch == 21:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    #if epoch == 46:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.00001

    for data in train_loader:
        data = data.cuda().to_variable(['input', 'target', 'pos'])
        data = transform(data)
        optimizer.zero_grad()
        class_loss, c_loss = model(data)
        loss = F.nll_loss(class_loss, data.target) + 0.1 * c_loss
        loss.backward()
        optimizer.step()
        #print(loss.data[0], c_loss.data[0])


def test(epoch, loader, dataset, str):
    model.eval()
    correct = 0
    c_loss_sum = 0
    for data in loader:
        data = data.cuda().to_variable(['input', 'pos'])
        data = transform(data)
        class_out, c_loss = model(data)
        pred = class_out.data.max(1)[1]
        correct += pred.eq(data.target).sum()
        c_loss_sum += c_loss.data[0]

    print('Epoch:', epoch, str, 'Accuracy:', correct / len(dataset), 'c_loss:',
          c_loss_sum / len(dataset))


for epoch in range(1, 31):
    train(epoch)
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, test_loader, test_dataset, 'Test')
