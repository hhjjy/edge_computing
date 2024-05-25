import os
import cv2
import numpy as np 
# 定義資料夾路徑
data_dir = './img_copy'
output_dir = './img_pre'
categories = ['1', '2', '3', '4']

# 確保輸出資料夾存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for category in categories:
    input_folder_path = os.path.join(data_dir, category)
    output_folder_path = os.path.join(output_dir, category)
    
    # 確保類別資料夾存在
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for img_name in os.listdir(input_folder_path):
        img_path = os.path.join(input_folder_path, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            # 圖像預處理
            img = cv2.resize(img, (64, 64))  # 調整圖像大小為64x64
            img = img / 255.0  # 將像素值歸一化到[0, 1]
            img = (img * 255).astype(np.uint8)  # 將像素值轉回0-255範圍
            
            output_img_path = os.path.join(output_folder_path, img_name)
            cv2.imwrite(output_img_path, img)
        else:
            print(f'Warning: Failed to read {img_path}')
