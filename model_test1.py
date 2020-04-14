import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



batch_size = 10240
num_filters = 16
depth = [32,48,64,80]
class MRCNN(nn.Module):
    def __init__(self):
        super(MRCNN,self).__init__()
        # input是(10240,1,400,4)
        self.conv1 = nn.Conv2d(1,num_filters,(1,4))
        # 完了是(10240,16,400,1)
        # (10240,16,20,20)
        self.conv2 = nn.Conv2d(num_filters,depth[0],3)
        #(10240,32,18,18)
        self.pool1 = nn.MaxPool2d(3,3)
        #(10240,32,6,6)
        self.conv3 = nn.Conv2d(depth[0],depth[1],3)
        #(10240,48,4,4)
        self.conv4 = nn.Conv2d(depth[1],depth[2],3)
        #(10240,64,2,2)
        self.fc1 = nn.Linear(depth[2]*2*2,10)
        self.fc2 = nn.Linear(10,1)
        self.sig = nn.Sigmoid()



    def forward(self,x):
        x= F.relu(self.conv1(x))
        x = x.reshape([-1,num_filters,20,20])
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1,depth[2]*2*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        x = self.sig(x)
        return x


use_cuda = torch.cuda.is_available()
num_epoch = 100
num_divide = 10
if __name__ == '__main__':

    if use_cuda:
        print('used cuda!')

    net = MRCNN()
    net = net.cuda() if use_cuda else net
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train(True)
    data = np.load('chr1.part1.npy')
    data = data.reshape([batch_size,1,400,4])
    target = np.load('lable_part1.npy')

    data,target = torch.FloatTensor(data),torch.FloatTensor(target)
    loss_all = []
    for epoch in range(num_epoch):
        data, target = Variable(data), Variable(target)
        if use_cuda:
            data,target = data.cuda(),target.cuda()
        #1024个放一批
        each = batch_size/num_divide
        for idx in range(num_divide):
            start = int(idx*each)
            end = int((idx+1)*each)
            output = net(data[start:end,:,:,:])
            output =output.reshape(int(each))
            loss = criterion(output, target[start:end])
            # print('loss is {}'.format(loss))
            loss_all.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    plt.figure(figsize=(10, 7))
    plt.plot(loss_all)  # record记载了每一个打印周期记录的训练和校验数据集上的准确度
    plt.xlabel('epoch/idx')
    plt.ylabel('loss rate')

    plt.show()



