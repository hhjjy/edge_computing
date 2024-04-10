'''
Author: Leo lion24161582@gmail.com
Date: 2024-04-09 22:21:07
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-04-10 23:31:14
FilePath: \edge_computing\HW3_B11002220\videoTest.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics import YOLO
import cv2
import math
import os
import threading
from config import *
# 初始化 YOLO 模型
classNames = ["hua", "leo"]
confidence_threshold = 0.8

def process_video(video_path, output_path, output_log_path, model_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 追蹤出現次數
    class_counts = {className: 0 for className in classNames}

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = math.ceil((box.conf[0]*100))/100
                if confidence < confidence_threshold:
                    continue
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cls = int(box.cls[0])
                class_counts[classNames[cls]] += 1
                cv2.putText(img, f"{classNames[cls]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        out.write(img)

    cap.release()
    out.release()
    with open(output_log_path, 'w') as log_file:
        for className, count in class_counts.items():
            log_file.write(f"{className}:{count}\n")
        log_file.write(f"FPS:{fps}\n")

def main():
    input_folder_path = TEST_VIDEO_FOLDER_PATH
    output_folder_path = OUTPUT_FOLDER_PATH
    weight_folder_path = WEIGHT_FOLDER_PATH

    # 獲取所有權重名稱（不含擴展名）
    weight_names = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(weight_folder_path) if f.endswith('.pt')]

    for weight_name in weight_names:
        model_path = os.path.join(weight_folder_path, weight_name + '.pt')  # 構造模型路徑

        output_weight_folder = os.path.join(output_folder_path, weight_name)
        
        # 檢查對應權重的輸出資料夾是否已存在
        if os.path.exists(output_weight_folder):
            print(f"Skipping weight {weight_name} as output folder already exists.")
            continue  # 如果存在，跳過此權重的處理

        # 如果輸出資料夾不存在，則創建資料夾
        os.makedirs(output_weight_folder)

        videos = [f for f in os.listdir(input_folder_path) if f.endswith('.mp4')]
        threads = []

        for video in videos:
            input_video_path = os.path.join(input_folder_path, video)
            output_video_path = os.path.join(output_weight_folder, video)
            output_log_path = os.path.join(output_weight_folder, os.path.splitext(video)[0] + "_result.txt")
            
            # 創建並啟動處理該視頻的線程
            t = threading.Thread(target=process_video, args=(input_video_path, output_video_path, output_log_path, model_path))
            t.start()
            threads.append(t)

        # 等待所有線程結束
        for t in threads:
            t.join()



if __name__ == "__main__":
    main()
