import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision import datasets, transforms
import time

# 创建神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output_layer = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        output = self.output_layer(x)
        return output

# 超参数
EPOCH = 2
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD = False  # 若已经下载mnist数据集则设为False

# 下载mnist数据
train_data = datasets.MNIST(
    root='./data',  # 保存路径
    train=True,  # True表示训练集，False表示测试集
    transform=transforms.ToTensor(),  # 将0~255压缩为0~1
    download=DOWNLOAD
)

# DataLoader
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_data = datasets.MNIST(
    root='./data',
    train=False
)

cnn = CNN()
# 将神经网络移到GPU上
cnn.cuda()
print(cnn)

# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# 损失函数
loss_func = nn.CrossEntropyLoss()

# 为了节约时间，只使用测试集的前2000个数据
test_x = Variable(
    torch.unsqueeze(test_data.data, dim=1),
    volatile=True
).type(torch.FloatTensor)[:2000] / 255  # 将将0~255压缩为0~1

test_y = test_data.targets[:2000]

# # 使用所有的测试集
# test_x = Variable(
#     torch.unsqueeze(test_data.test_data, dim=1),
#     volatile=True
# ).type(torch.FloatTensor)/255 # 将将0~255压缩为0~1

# test_y = test_data.test_labels

# 将测试数据移到GPU上
test_x = test_x.cuda()
test_y = test_y.cuda()

# 开始计时
start = time.time()

# 训练神经网络
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        # 将训练数据移到GPU上
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = cnn(test_x)
            # 将预测结果移到GPU上
            predict_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = (predict_y == test_y).sum().item() / test_y.size(0)
            print('Epoch', epoch, '|', 'Step', step, '|', 'Loss', loss.data.item(), '|', 'Test Accuracy', accuracy)

end = time.time()

# 训练耗时
print('Time cost:', end - start, 's')

# 预测
test_output = cnn(test_x[:100])
# 为了将CUDA tensor转化为numpy，需要将数据移回CPU上
# 否则会报错：TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
predict_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()
real_y = test_y[:100].cpu().numpy()
print(predict_y)
print(real_y)

# 打印预测和实际结果
for i in range(10):
    print('Predict', predict_y[i])
    print('Real', real_y[i])
