'''
Author: Leo lion24161582@gmail.com
Date: 2024-03-14 22:38:42
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-03-14 22:38:48
FilePath: \HW2_B11002220\onnxinfer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
    transform = transforms.Compose([#創建一個轉換序列
        transforms.Resize((32, 32)),#將圖像大小調整為32x32像素。
        transforms.ToTensor(),#將Pillow圖像或NumPy ndarray轉換為torch.Tensor。這也將圖像的像素值從0-255縮放到0-1之間。
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#標準化圖像的每個通道。給定的均值（第一個三元組）和標準差（第二個三元組）用於將像素值進一步縮放。這裡，它將0-1範圍的像素值轉換為-1到1。
    ])
    image = transform(image).unsqueeze(0)
    return image
def main():
    parser = argparse.ArgumentParser(description='ONNX CIFAR-10 Inference')
    parser.add_argument('--img', type=str, required=True, help='path to the input image')
    args = parser.parse_args()

    # 載入 ONNX 模型並指定 GPU 運行
    session = onnxruntime.InferenceSession("./cifar10.onnx", providers=['CUDAExecutionProvider'])

    # 讀取並預處理圖片
    image = load_image(args.img).to("cuda") 
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
