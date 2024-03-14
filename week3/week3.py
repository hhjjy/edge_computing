# Modified from PyTorch examples:
# https://github.com/pytorch/examples/blob/master/cifar10/main.py
#
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_B0_Weights

# class Net(nn.Module):
#     def __init__(self, num_classes=10):
#         super(Net, self).__init__()
#         # 加载预训练的EfficientNet-B0模型
#         self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

#         # 替换EfficientNet的分类器以匹配CIFAR-10的类别数
#         num_ftrs = self.efficientnet.classifier[1].in_features
#         self.efficientnet.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(num_ftrs, num_classes),
#         )
        
#         # EfficientNet是为较大的图像设计的，但它可以通过调整全局平均池化层后的线性层来适应较小的图像
#         # 这里不需要修改第一个卷积层或移除最大池化层

#     def forward(self, x):
#         return self.efficientnet(x)
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # 加载预训练的ResNet模型
        self.resnet = models.resnet18(pretrained=True)
    
        # 替换ResNet的全连接层以匹配CIFAR-10的类别数
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
        # 替换ResNet的第一个卷积层，以适应CIFAR-10图像的尺寸
        # CIFAR-10图像较小，因此使用较小的卷积核和步长
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # 移除最大池化层

    def forward(self, x):
        return self.resnet(x)
    
# class Ｎet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(Net, self).__init__()
#         # 加载预训练的MobileNetV2模型
#         self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAUL)
#         # 替换分类器以匹配CIFAR-10的类别数
#         num_ftrs = self.mobilenet.classifier[1].in_features
#         self.mobilenet.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(num_ftrs, num_classes),
#         )

#     def forward(self, x):
#         return self.mobilenet(x)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(12544, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # 訓練參數
    parser = argparse.ArgumentParser(description='PyTorch MNIST Predictor')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # if use_cuda:
        # device = torch.device("cuda")
    device = torch.device("mps")

    # else:
        # device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 載入訓練集及測試集
    dataset1 = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    # 載入模型並開始訓練
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    # 儲存模型
    torch.save(model.state_dict(), "cifar10_model.pt")

if __name__ == '__main__':
    main()
