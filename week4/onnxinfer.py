import onnxruntime
from torchvision import transforms, models
from PIL import Image
import numpy as np  
import torch.nn.functional as F
import torch
import argparse
import time

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
    parser = argparse.ArgumentParser(description='ONNX CIFAR-10 Inference')
    parser.add_argument('--img', type=str, required=True, help='path to the input image')
    args = parser.parse_args()

    # 載入 ONNX 模型並指定 GPU 運行
    session = onnxruntime.InferenceSession("./cifar10.onnx", providers=['CoreMLExecutionProvider'])

    # 讀取並預處理圖片
    image = load_image(args.img).to("mps") 
    input_dict = {session.get_inputs()[0].name: image.cpu().numpy()}

    # 進行推理
    t0 = time.time()
    for i in range(10000):
        output = session.run(None, input_dict)
    t1 = time.time()

    # 將原始分數轉換為概率
    probabilities = F.softmax(torch.tensor(output[0]), dim=1)

    # 獲取最大概率對應的類別
    predicted_class = torch.argmax(probabilities, dim=1).item()
    labels = ["飛機", "汽車", "鳥", "貓", "鹿", "狗", "青蛙", "馬", "船", "卡車"]
    print("概率分佈：", probabilities.numpy())
    print("預測類別：", labels[predicted_class])
    print('ONNX推論10000次消耗時間', int(t1-t0), 's')

if __name__ == '__main__':
    main()
