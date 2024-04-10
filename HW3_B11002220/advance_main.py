'''
Author: Leo lion24161582@gmail.com
Date: 2024-04-10 00:20:56
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-04-10 00:42:18
FilePath: \edge_computing\HW3_B11002220\advance_main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from multiprocessing import Process, Queue,Event
import cv2
from ultralytics import YOLO
import time
import math
from config import *
classNames = ["hua", "leo"]
confidence_threshold = 0.8  # 信心閥值

def detection_process(image_queue, detection_queue, stop_event):
    # MODEL_PATH = "best_80_v5.pt"
    model = YOLO(MODEL_PATH)
    while not stop_event.is_set():
        if not image_queue.empty():
            img = image_queue.get()
            results = model(img, stream=True)
            detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    if confidence < confidence_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, confidence, int(box.cls[0])))

            detection_queue.put(detections)
        else:
            time.sleep(0.01)  # 暫停一下以減少忙等待
    print("Detection process ended.")
def display_process(image_queue, detection_queue, stop_event):
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    display_duration = 2  # 每次檢測結果顯示持續時間（秒）
    last_detection_time = None
    detections = []

    while not stop_event.is_set():
        if not image_queue.empty():
            img = image_queue.get()

        if not detection_queue.empty():
            detections = detection_queue.get()
            last_detection_time = time.time()

        # 如果有檢測結果且未超過顯示持續時間
        if last_detection_time and time.time() - last_detection_time < display_duration:
            for det in detections:
                x1, y1, x2, y2, confidence, cls = det
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, f"{classNames[cls]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Display process ended.")

if __name__ == '__main__':
    image_queue = Queue()
    detection_queue = Queue()
    stop_event = Event()
    
    display_p = Process(target=display_process, args=(image_queue, detection_queue, stop_event))
    detection_p = Process(target=detection_process, args=(image_queue, detection_queue, stop_event))
    
    try:
        display_p.start()
        detection_p.start()

        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        while not stop_event.is_set():
            success, img = cap.read()
            if success:
                image_queue.put(img)
            else:
                stop_event.set()
        
        # 等待直到用戶按下 Ctrl+C
    except KeyboardInterrupt:
        print("Interrupted! Cleaning up...")
        stop_event.set()
    finally:
        # 釋放資源
        cap.release()
        
        # 安全地結束子進程
        detection_p.terminate()
        display_p.terminate()
        detection_p.join()
        display_p.join()
        
        print("Resources have been released and processes have been terminated.")