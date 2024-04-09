from ultralytics import YOLO
import cv2
import math
import time

# 輸入和輸出檔案名稱
input_video_path = 'input.mp4'
output_video_path = 'output.mp4'

# 初始化 YOLO 模型
MODEL_PATH = "./best_80_v5.pt"
model = YOLO(MODEL_PATH)
classNames = ["hua", "leo"]

# 打開輸入影片
cap = cv2.VideoCapture(input_video_path)

# 獲取影片的基本資訊
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 定義輸出影片的編碼和檔案
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定義編碼格式
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

confidence_threshold = 0.8  # 信心閥值

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break  # 如果讀取失敗或影片結束，跳出循環

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

    # 將處理後的影像寫入輸出檔案
    out.write(img)

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
