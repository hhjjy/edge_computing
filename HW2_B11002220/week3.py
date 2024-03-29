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

# 定義網絡結構
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 使用預訓練的ResNet18
        num_ftrs = self.resnet.fc.in_features  # 獲取全連接層的輸入特徵數
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # 由於原始的 ResNet18 是針對 ImageNet 數據集預訓練的，其全連接層的輸出維度對應於 ImageNet 的類別數（1000類）。對於 CIFAR-10 數據集（10類），需要將這個全連接層替換爲一個新的線性層，其輸出維度爲 CIFAR-10 的類別數。
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)# 修改第一層捲機層以讓他適應新的輸入
        self.resnet.maxpool = nn.Identity()  # 移除最大池化層，保留原始細節。

    def forward(self, x):
        return self.resnet(x)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()# 告訴硬體開始訓練 train表示可能會頻繁的更動參數
    for batch_idx, (data, target) in enumerate(train_loader):# trainloader 會把資料切成一筆一筆
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
    model.eval()# 將模型設置爲評估模式。這會通知所有層在評估時應禁用特定行爲，如 dropout 層的隨機失活。
    test_loss = 0
    correct = 0
    with torch.no_grad():#禁止某評估計算時儲存梯度，以節省記憶體與计算资源
        for data, target in test_loader:#載入標籤與資料
            data, target = data.to(device), target.to(device) #送入硬體做處理
            output = model(data) #經過計算取得節骨
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # 計算Loss 大小 loss 越小表示預測的與實際更近
            pred = output.argmax(dim=1, keepdim=True)# 從一維的標籤中尋找概率最高的作為標籤輸入
            correct += pred.eq(target.view_as(pred)).sum().item()# 增加正確數量

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
                        help='learning rate (default: 1.0)')#學習率設定
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
    # 取得額外添加的參數 透過namespace類別儲存。Namespace(batch_size=256, test_batch_size=1000, epochs=10, lr=0.001, gamma=0.7, no_cuda=False, dry_run=False, seed=1, log_interval=10)
 
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()#只有當no cuda為false且cuda是可用的才會設置為true

    torch.manual_seed(args.seed) # 設定一個亂數種子1 ，好處是可以避免
    
    if use_cuda:# 使用cuda
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size} # 設定成測試集的大小
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)#把新的設定更新倒train_args 內  {'num_workers': 1,'pin_memory': True,'shuffle': True }
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
    # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,batch_sampler=None, num_workers=0, collate_fn=None,pin_memory=False, drop_last=False, timeout=0,worker_init_fn=None, *, prefetch_factor=2,persistent_workers=False)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)# 把對應的參數對過關鍵字解包的方式載入
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    # 載入模型並開始訓練
    model = Net().to(device)#模型載入到cuda
    optimizer = optim.Adam(model.parameters(), lr=args.lr)#優化器是根據模型在訓練數據上的表現（通常是損失函數的值），來更新和調整模型的參數，以便減少損失函數的值，從而提高模型的準確性。

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)#Scheduler 通過預定的策略在訓練過程中動態調整學習率，例如隨著訓練進展逐步降低學習率，以幫助模型更細緻地接近全局最小損失。
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)# 訓練一次模型
        test(model, device, test_loader)# 測試模型輸出
        scheduler.step()# 學習率調整
    # 儲存模型
    torch.save(model.state_dict(), "cifar10_model.pt")

if __name__ == '__main__':
    main()
