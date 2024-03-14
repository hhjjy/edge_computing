import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import onnx

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
    
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  

    def forward(self, x):
        return self.resnet(x)


package_dir = os.path.dirname(os.path.abspath(__file__))
default_model_path = os.path.join(package_dir, 'cifar10_model.pt')

parser = argparse.ArgumentParser(description='PyTorch cifar10 Predictor')
parser.add_argument('--model', type=str, default=default_model_path,
                    help='model for prediction (default: {})'.format(default_model_path))
args = parser.parse_args()

# 使用指定的模型檔案路徑
model = Net()
model_r = torch.load(default_model_path, map_location="cuda")
model.load_state_dict(model_r)
model.eval()

x = torch.randn(1, 3, 32, 32,  requires_grad=True)
torch.onnx.export(model, x, 'cifar10.onnx', input_names=['input'], output_names=['output'], verbose=False)
