'''
Author: Leo lion24161582@gmail.com
Date: 2024-03-14 22:38:22
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-03-14 22:38:27
FilePath: \HW2_B11002220\export.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import onnx
# 定義網絡結構
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 使用預訓練的ResNet18
        num_ftrs = self.resnet.fc.in_features  # 獲取全連接層的輸入特徵數
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # 由於原始的 ResNet18 是針對 ImageNet 數據集預訓練的，其全連接層的輸出維度對應於 ImageNet 的類別數（1000類）。對於 CIFAR-10 數據集（10類），需要將這個全連接層替換爲一個新的線性層，其輸出維度爲 CIFAR-10 的類別數。
        # 修改第一層卷積層和移除最大池化層
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)# 修改第一層捲機層以讓他適應新的輸入
        self.resnet.maxpool = nn.Identity()  # 移除最大池化層，保留原始細節。

    def forward(self, x):
        return self.resnet(x)
#取得存這個檔案的資料夾名稱
package_dir = os.path.dirname(os.path.abspath(__file__))
default_model_path = os.path.join(package_dir, 'cifar10_model.pt')# 把路徑與模型組合成模型絕對路徑
parser = argparse.ArgumentParser(description='PyTorch cifar10 Predictor') # 參數設定
parser.add_argument('--model', type=str, default=default_model_path,
                    help='model for prediction (default: {})'.format(default_model_path))#參數設定type
args = parser.parse_args()
# 使用指定的模型檔案路徑
model = Net()#採用Net類別的模型
model_r = torch.load(default_model_path, map_location="cuda")#
model.load_state_dict(model_r)#保存
model.eval()

x = torch.randn(1, 3, 32, 32,  requires_grad=True)
torch.onnx.export(model, x, 'cifar10.onnx', input_names=['input'], output_names=['output'], verbose=False)
