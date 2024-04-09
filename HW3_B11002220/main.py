from ultralytics import YOLO
import cv2
import math 
import time 
# from config import *
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
MODEL_PATH = "best_80_v5.pt"
# 模型
model = YOLO(MODEL_PATH)
classNames = ["hua", "leo"]

frame_counter = 0
total_time = 0  
paused = False
confidence_threshold = 0.8  # 信心閥值

while True:
    if not paused:
        start_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = math.ceil((box.conf[0]*100))/100
                if confidence < confidence_threshold:
                    continue  # 如果閥值低於跳過
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cls = int(box.cls[0])
                org = [x1, y1 - 10]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5  
                color = (255, 255, 0)  
                thickness = 2
                cv2.putText(img, f"{classNames[cls]} {confidence:.2f}", org, font, fontScale, color, thickness)

                print(f"Detected: {classNames[cls]} with confidence: {confidence:.2f}")
        
        # FPS計算
        end_time = time.time()
        frame_time = end_time - start_time
        # 顯示FPS和平均FPS
        cv2.putText(img, f"Current FPS: {1/frame_time:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Webcam', img)
    
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"captured_frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, img)
        print(f"Image saved as {filename}")
    elif key == ord('p'):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
