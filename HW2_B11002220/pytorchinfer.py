'''
Author: Leo lion24161582@gmail.com
Date: 2024-03-14 22:39:09
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-03-14 22:39:14
FilePath: \HW2_B11002220\pytorchinfer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import argparse
import time

# 定義網絡結構
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 使用預訓練的ResNet18
        num_ftrs = self.resnet.fc.in_features  # 獲取全連接層的輸入特徵數
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # 替換全連接層
        # 修改第一層卷積層和移除最大池化層
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # 移除最大池化層

    def forward(self, x):
        return self.resnet(x)
        
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),#將圖像大小調整為32x32像素。
        transforms.ToTensor(),#將Pillow圖像或NumPy ndarray轉換為torch.Tensor。這也將圖像的像素值從0-255縮放到0-1之間。
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    return image

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Inference')
    parser.add_argument('--img', type=str, required=True, help='path to the input image')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load("cifar10_model.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        image = load_image(args.img).to(device) 
        # 進行推理
        t0 = time.time()
        for i in range(10000):
            output = model(image)
        t1 = time.time()
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    labels = ["飛機", "汽車", "鳥", "貓", "鹿", "狗", "青蛙", "馬", "船", "卡車"]
    print("概率分佈：", probabilities)
    print("預測類別：", labels[predicted_class])
    print('PyTorch推論10000次消耗時間', int(t1-t0), 's')

if __name__ == '__main__':
    main()
