import os
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')



from torch_geometric.datasets import ModelNet10RandAugPC, ModelNet10  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import (NormalizeScale, RandomFlip,
                                       CartesianAdj, RandomTranslate,
                                       PCToGraphNN, PCToGraphRad, LogCartesianAdj,
                                       SamplePointcloud, RandomRotate,
                                       CartesianSphereAdj)  # noqa
from torch_geometric.nn.modules import SplineConv, QuatCapsuleLayer, \
    PoolingByQuatAgreement  # noqa
from torch.nn import Linear as Lin # noqa
from torch_geometric.nn.functional import (sparse_voxel_max_pool,
                                           dense_voxel_max_pool,
                                           sparse_voxel_avg_pool,
                                           dense_voxel_avg_pool)  # noqa
from torch_geometric.visualization.model import show_model  # noqa

path = os.path.dirname(os.path.realpath(__file__))


# Pointcloud Version
path = os.path.join(path, '..', 'data', 'ModelNet10RandAugPC')
to_graph = PCToGraphNN()
#to_graph = PCToGraphRad(r=0.03)
transform = CartesianSphereAdj()
#init_transform2 = CartesianAdj(r=0.03)
init_transform = Compose([SamplePointcloud(num_points=2048), NormalizeScale(),
                          to_graph])
init_transform_rot = Compose([SamplePointcloud(num_points=2048), NormalizeScale(),
                              RandomRotate(np.pi), to_graph])
train_dataset = ModelNet10RandAugPC(path, True, transform=init_transform)
test_dataset = ModelNet10RandAugPC(path, False, transform=init_transform)
test_dataset_rot = ModelNet10RandAugPC(path, False, transform=init_transform_rot)
batch_size = 6
num_classes = 10
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size)
test_loader_rot = DataLoader2(test_dataset_rot, batch_size=batch_size)
'''

# Mesh Version
path = os.path.join(path, '..', 'data', 'ModelNet10')
transform = LogCartesianAdj(30)
init_transform = NormalizeScale()
train_dataset = ModelNet10(path, True, init_transform)
test_dataset = ModelNet10(path, False, init_transform)
batch_size = 3
num_classes = 10
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size)
'''




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = SplineConv(1, 24, dim=3, kernel_size=5)
        #self.conv2 = SplineConv(36, 36, dim=3, kernel_size=5)
        #self.conv3 = SplineConv(48, 48, dim=3, kernel_size=5)
        #self.conv4 = SplineConv(60, 60, dim=3, kernel_size=5)
        #self.conv5 = SplineConv(72, 72, dim=3, kernel_size=5)
        #self.conv2 = SplineConv(6, 6, dim=3, kernel_size=5)
        #self.conv3 = SplineConv(8, 8, dim=3, kernel_size=5)
        #self.conv4 = SplineConv(10, 10, dim=3, kernel_size=5)
        #self.conv5 = SplineConv(12, 12, dim=3, kernel_size=5)
        #self.conv1 = MLPConv(1, 24, dim=3)
        #self.conv2 = MLPConv(36, 36, dim=3)
        #self.conv3 = MLPConv(48, 48, dim=3)
        #self.conv4 = MLPConv(60, 60, dim=3)
        #self.conv5 = MLPConv(72, 72, dim=3)


        self.conv1 = SplineConv(1, 16, dim=3, kernel_size=5)
        self.lin = Lin(16, 4)
        self.caps0 = QuatCapsuleLayer(num_caps_in=1, num_caps_out=6)
        self.caps1 = QuatCapsuleLayer(num_caps_in=6, num_caps_out=6)
        self.pool1 = PoolingByQuatAgreement(num_caps_in=6, num_caps_out=6)
        self.caps2 = QuatCapsuleLayer(num_caps_in=6, num_caps_out=6)
        self.pool2 = PoolingByQuatAgreement(num_caps_in=6, num_caps_out=6)
        self.caps3 = QuatCapsuleLayer(num_caps_in=6, num_caps_out=8)
        self.pool3 = PoolingByQuatAgreement(num_caps_in=8, num_caps_out=8)
        self.caps4 = QuatCapsuleLayer(num_caps_in=8, num_caps_out=8)
        self.pool4 = PoolingByQuatAgreement(num_caps_in=8, num_caps_out=8)
        self.caps5 = QuatCapsuleLayer(num_caps_in=8, num_caps_out=12)
        self.pool5 = PoolingByQuatAgreement(num_caps_in=12, num_caps_out=12)
        self.caps6 = QuatCapsuleLayer(num_caps_in=12, num_caps_out=14)
        self.pool6 = PoolingByQuatAgreement(num_caps_in=14, num_caps_out=14)
        self.caps7 = QuatCapsuleLayer(num_caps_in=14, num_caps_out=num_classes,
                                      use_conv=False)

        self.fc1 = Lin(16,64)
        self.fc2 = Lin(64,10)


    def pool_args(self, mean, x, min_pos, max_pos):
        if not self.training:
            return (max_pos+0.001-min_pos) / mean, min_pos
        #size = [(max_pos+0.001-min_pos) / random.uniform(mean - x, mean + x) for _ in range(3)]
        size = (max_pos+0.001-min_pos) / mean
        #start = [min_pos + random.uniform(-size[s], 0) for s in range(3)]
        start = min_pos
        return size, start

    def build_pose_vectors(self, num_caps, input=None, size=0):
        if input is not None:
            #zeros = Variable(torch.zeros(3).view(1, 1, 1, 3).
            #                 expand(input.size(0), num_caps, -1, -1).cuda())
            one = Variable(torch.zeros(4).view(1, 1, 1, 4).
                           expand(input.size(0), num_caps, -1, -1).cuda())
            one[:, :, :, 3] = 1.0
            #temp = torch.cat([input.view(-1, num_caps, 3, 3), zeros], dim=2)
            pose_vectors = torch.cat([input.view(-1, num_caps, 3, 4), one], dim=2)
        else:
            zeros = torch.zeros(4, 4)
            zeros.copy_(torch.eye(4))
            pose_vectors = zeros.view(1,1,4,4).expand(size, num_caps, -1, -1)
            pose_vectors = Variable(pose_vectors.cuda())
        return pose_vectors

    def normalize(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        return tensor / (torch.sqrt(squared_norm)+0.0001)

    def forward(self, data):

        quats = self.lin(F.elu(self.conv1(data.adj, data.input)))
        data.pose_vectors = self.normalize(quats.view(-1,
                                                      self.caps0.num_caps_in,
                                                      4))
        data.input = data.input.view(-1, 1).expand(-1, self.caps0.num_caps_in).\
            contiguous()
        data = self.caps0(data)
        data = self.caps1(data)
        min_pos, max_pos = data.pos.min(), data.pos.max()

        size, start = self.pool_args(32, 8, min_pos, max_pos)
        data, _ = self.pool1(data, size, start, transform)

        #data = self.caps2(data)

        #size, start = self.pool_args(16, 2, min_pos, max_pos)
        #data, _ = self.pool2(data, size, start, transform)

        data = self.caps3(data)

        size, start = self.pool_args(8, 2, min_pos, max_pos)
        data, _ = self.pool3(data, size, start, transform)

        #data = self.caps4(data)

        #size, start = self.pool_args(4, 1, min_pos, max_pos)
        #data, _ = self.pool4(data, size, start, transform)

        data = self.caps5(data)

        size, start = self.pool_args(2, 1, min_pos, max_pos)
        data, _ = self.pool5(data, size, start, transform)

        data = self.caps6(data)

        data, _ = self.pool6(data, max_pos-min_pos+1, min_pos-0.5, None)
        data = self.caps7(data)

        classes = data.input
        #print(classes.data[0])
        return classes
        #return F.softmax(classes, dim=-1)
        #return F.log_softmax(classes, dim=1)

model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def class_loss(labels, classes):
    left = F.relu(0.9 - classes, inplace=True) ** 2
    right = F.relu(classes - 0.1, inplace=True) ** 2

    margin_loss = labels * left + 0.5 * (1. - labels) * right
    margin_loss = margin_loss.sum(1).mean()

    return margin_loss

def spread_loss(labels, classes, epoch):
    m = Variable(torch.FloatTensor(1).fill_(min(0.1+0.1*epoch,0.9)).cuda(), requires_grad=False)
    act_t = (classes*labels).sum(1)
    l = ((F.relu(m - (act_t.view(-1,1) - classes))**2)*(1-labels)).sum(1).mean()
    return l


def train(epoch):
    model.train()


    #if epoch == 6:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.0005

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.003

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    #if epoch == 21:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.0001

    loss_sum = 0
    loss_count = 0
    for data in train_loader:
        data = data.cuda().to_variable(['input', 'target'])
        data = transform(data)
        optimizer.zero_grad()
        classes = model(data)
        onehot = torch.zeros(classes.size(0),num_classes).cuda()

        onehot.scatter_(1, data.target.data.view(-1,1).expand(-1,num_classes),1)
        onehot = Variable(onehot)
        #loss = class_loss(onehot, classes)
        loss = spread_loss(onehot, classes, epoch)
        #loss = F.nll_loss(classes, data.target)
        if np.any(np.isnan(classes.data.cpu().numpy())):
            print('Error: NaN!')
        else:
            loss.backward()
            optimizer.step()
            loss_sum += loss.data[0]
            loss_count += 1
            if loss_count == 30:
                print(loss_sum/loss_count)
                loss_sum = 0
                loss_count = 0



def test(epoch, loader, dataset, str):
    model.eval()
    correct = 0
    false_per_class = np.zeros(10)
    examples_per_class = np.zeros(10)
    for data in loader:
        data = data.cuda().to_variable(['input'])
        data = transform(data)
        pred = model(data)
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



    accuracy_per_class = (examples_per_class-false_per_class)/examples_per_class
    np.set_printoptions(precision=4)
    print('Epoch:', epoch, str, 'Accuracy:', correct / len(dataset))
    print(accuracy_per_class, 'Class Accuracy:',accuracy_per_class.mean())
    print(false_per_class)



print('Train 3dCapsuleNet')
for epoch in range(1, 31):
    train(epoch)
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, test_loader, test_dataset, 'Test')
    test(epoch, test_loader_rot, test_dataset_rot, 'Test_rot')