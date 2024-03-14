
import cv2
def face_detection(frame):
    """
    進行人臉檢測
    :param frame: 輸入的影像幀
    :return: 檢測到的人臉座標表
    """
    # 加載OpenCV預訓練的人臉檢測器
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
    # 調整影像大小，使其寬度或高度不超過1080
    max_dimension = 1080
    height, width = frame.shape[:2]
    if height > max_dimension or width > max_dimension:
        scaling_factor = max_dimension / max(height, width)
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # 將影像轉為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 在灰階影像中進行人臉檢測
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def image_face_detection(image_path):
    """
    對指定路徑的影像進行人臉檢測
    :param image_path: 影像的路徑
    """
    # 讀取影像
    image = cv2.imread(image_path)
    # 進行人臉檢測
    faces = face_detection(image)
    # 在檢測到的人臉周圍畫矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 顯示檢測結果
    cv2.imshow('Face Detection - Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 初始化Camera
    video_capture = cv2.VideoCapture(0)

    # 使用Camera進行實時人臉檢測
    while True:
        result, video_frame = video_capture.read()  # 從Camera讀取Frame
        if result:
            faces = face_detection(video_frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.imshow("My Face Detection Project", video_frame)  # 顯示處理後的Frame
            #等待一毫秒 如果按下Q就離開
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_capture.release()
    cv2.destroyAllWindows()