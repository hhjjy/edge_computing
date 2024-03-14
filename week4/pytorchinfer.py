import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import argparse
import time

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  
    def forward(self, x):
        return self.resnet(x)

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    return image

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Inference')
    parser.add_argument('--img', type=str, required=True, help='path to the input image')
    args = parser.parse_args()
    device = torch.device("mps")
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
