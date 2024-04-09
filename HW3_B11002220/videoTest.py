from ultralytics import YOLO
import cv2
import math
import os
import threading
from config import *
# 初始化 YOLO 模型
model = YOLO(MODEL_PATH)
classNames = ["hua", "leo"]
confidence_threshold = 0.8

def process_video(video_path, output_path, output_log_path):
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
            log_file.write(f"{className}:{count}次\n")
        log_file.write(f"FPS:{fps}\n")

def main():
    input_folder_path = 'input'
    output_folder_path = 'output'
    weight_name = os.path.basename(MODEL_PATH).split('.')[0]

    output_weight_folder = os.path.join(output_folder_path, weight_name)
    if not os.path.exists(output_weight_folder):
        os.makedirs(output_weight_folder)

    videos = [f for f in os.listdir(input_folder_path) if f.endswith('.mp4')]
    threads = []

    for video in videos:
        input_video_path = os.path.join(input_folder_path, video)
        output_video_path = os.path.join(output_weight_folder, video)
        output_log_path = os.path.join(output_weight_folder, os.path.splitext(video)[0] + "_result.txt")  # 结果日志路径
        t = threading.Thread(target=process_video, args=(input_video_path, output_video_path, output_log_path))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
