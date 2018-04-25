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



from torch_geometric.datasets import ModelNet10RandAugPC  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import (NormalizeScale, RandomFlip,
                                       CartesianAdj, RandomTranslate,
                                       PCToGraphNN, PCToGraphRad,
                                       SamplePointcloud, RandomRotate)  # noqa
from torch_geometric.nn.modules import InvGraphConv  # noqa
from torch.nn import Linear as Lin # noqa
from torch_geometric.nn.functional import (sparse_voxel_max_pool,
                                           dense_voxel_max_pool,
                                           sparse_voxel_avg_pool,
                                           dense_voxel_avg_pool)  # noqa
from torch_geometric.visualization.model import show_model  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10RandAugPC')

to_graph = PCToGraphNN()
#to_graph = PCToGraphRad(r=0.03)
transform = CartesianAdj()
init_transform = Compose([SamplePointcloud(num_points=2048), NormalizeScale(),
                          to_graph])
train_dataset = ModelNet10RandAugPC(path, True, transform=init_transform)
test_dataset = ModelNet10RandAugPC(path, False, transform=init_transform)
batch_size = 6
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = InvGraphConv(1, 16, dim=3, kernel_size=5, num_conv=2)
        self.conv2 = InvGraphConv(19, 16, dim=3, kernel_size=5, num_conv=2)
        self.conv3 = InvGraphConv(19, 24, dim=3, kernel_size=5, num_conv=2)
        self.conv4 = InvGraphConv(27, 32, dim=3, kernel_size=5, num_conv=2)
        self.conv5 = InvGraphConv(35, 40, dim=3, kernel_size=5, num_conv=2)
        self.fc1 = nn.Linear(8 * 43, 10)

        self.lin1 = Lin(19, 19)
        self.lin2 = Lin(19, 19)
        self.lin3 = Lin(27, 27)
        self.lin4 = Lin(35, 35)
        self.lin5 = Lin(43, 43)

    def pool_args(self, mean, x):
        if not self.training:
            return 1 / mean, 0
        #size = [1 / random.uniform(mean - x, mean + x) for _ in range(3)]
        size = 1 / mean
        #start = [random.uniform(-1 / (mean - x), 0) for _ in range(3)]
        start = 0
        return size, start

    def forward(self, data):
        data.input, c_loss1 = self.conv1(data.adj, data.input)
        data.input = F.elu(self.lin1(data.input))
        #att1 = self.att1(data.input)
        size, start = self.pool_args(32, 8)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input, c_loss2 = self.conv2(data.adj, data.input)
        data.input = F.elu(self.lin2(data.input))
        #att2 = self.att2(data.input)
        size, start = self.pool_args(16, 4)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input, c_loss3 = self.conv3(data.adj, data.input)
        data.input = F.elu(self.lin3(data.input))
        #att3 = self.att3(data.input)
        size, start = self.pool_args(8, 2)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input, c_loss4 = self.conv4(data.adj, data.input)
        data.input = F.elu(self.lin4(data.input))
        #att4 = self.att4(data.input)
        size, start = self.pool_args(4, 1)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input, c_loss5 = self.conv5(data.adj, data.input)
        data.input = F.elu(self.lin5(data.input))
        data, _ = dense_voxel_max_pool(data, 1, -0.5, 1.5)

        x = data.input.view(-1, self.fc1.weight.size(1))
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1), c_loss1 + c_loss2 + c_loss3 + \
               c_loss4 + c_loss5


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer_pca = torch.optim.Adam(model.parameters(), lr=0.01)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()

    if epoch == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01

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
        loss = F.nll_loss(class_loss, data.target)
        loss.backward()
        #c_loss.backward()
        optimizer.step()
        if model.conv1.local_stn1.weight.grad.data.max() != 0:
            print(model.conv1.local_stn1.weight.grad.data.max())
        #print(loss.data[0], c_loss.data[0])


def train_pcanet(epoch):
    model.train()
    if epoch == 1:
        for param_group in optimizer_pca.param_groups:
            param_group['lr'] = 0.01

    if epoch == 8:
        for param_group in optimizer_pca.param_groups:
            param_group['lr'] = 0.001

    if epoch == 9:
        for param_group in optimizer_pca.param_groups:
            param_group['lr'] = 0.0001

    loss_sum = 0
    loss_count = 0
    for i,data in enumerate(train_loader):
        data = data.cuda().to_variable(['input', 'target', 'pos'])
        data = transform(data)
        optimizer_pca.zero_grad()
        class_loss, c_loss = model(data)
        loss = c_loss
        loss.backward()
        loss_sum += loss.data[0]
        loss_count += batch_size
        optimizer_pca.step()

        if i%200 == 199:
            print(epoch, i+1, loss_sum/loss_count)
            loss_count = 0
            loss_sum = 0


def test(epoch, loader, dataset, str):
    model.eval()
    correct = 0
    false_per_class = np.zeros(10)
    examples_per_class = np.zeros(10)
    c_loss_sum = 0
    for data in loader:
        data = data.cuda().to_variable(['input', 'pos'])
        data = transform(data)
        pred, c_loss = model(data)
        pred = pred.data.max(1)[1]
        eq = pred.eq(data.target).cpu()
        false = (1-eq).nonzero()
        false = false.view(-1)
        if false.dim() > 0:
            uniques, fpc = np.unique(data.target.cpu()[false].numpy(),
                               return_counts=True)
            false_per_class[uniques] += fpc
        uniques_epc, epc = np.unique(data.target.cpu().numpy(),
                                 return_counts=True)
        examples_per_class[uniques_epc] += epc
        correct += eq.sum()
        c_loss_sum += c_loss.data[0]



    accuracy_per_class = (examples_per_class-false_per_class)/examples_per_class
    np.set_printoptions(precision=4)
    print('Epoch:', epoch, str, 'Accuracy:', correct / len(dataset), 'c_loss:',
          c_loss_sum / len(dataset))
    print(accuracy_per_class, 'Class Accuracy:',accuracy_per_class.mean())
    print(false_per_class)


#test(0, train_loader, train_dataset, 'Train')
#test(0, test_loader, test_dataset, 'Test')
print('Train PCANet')
for epoch in range(1,10):
    train_pcanet(epoch)

print('Train 3dCapsuleNet')
for epoch in range(1, 31):
    train(epoch)
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, test_loader, test_dataset, 'Test')