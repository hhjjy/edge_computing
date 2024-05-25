'''
Author: Leo lion24161582@gmail.com
Date: 2024-05-25 17:03:22
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-05-25 17:03:29
FilePath: \期末\標註\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2

def generate_filename(theme, count):
    return f"{theme}_{count:04d}.jpg"

def capture_images():
    # 定義主題
    theme = "4"
    count = 0  # 初始化計數器
    
    # 打開USB攝像頭
    cap = cv2.VideoCapture(0)
    
    # 設置攝像頭解析度為640x640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    # 檢查攝像頭是否成功打開
    if not cap.isOpened():
        print("無法打開攝像頭。")
        return
    
    while True:
        # 讀取圖像幀
        ret, frame = cap.read()
        
        # 如果無法讀取圖像，則退出迴圈
        if not ret:
            print("無法讀取圖像。")
            break
        
        try:
            # 顯示圖像
            cv2.imshow('Image', frame)
        except cv2.error as e:
            print(f"OpenCV錯誤: {e}")
            break
        
        # 等待按鍵s的按下，按下s後採集圖片
        key = cv2.waitKey(1)
        if key == ord('s'):
            # 生成唯一的文件名
            file_name = generate_filename(theme, count)
            count += 1  # 增加計數器
            
            # 保存圖片
            cv2.imwrite(file_name, frame)
            print(f"已捕獲圖片，保存為{file_name}。")
        
        # 按下q退出迴圈
        elif key == ord('q'):
            break
    
    # 釋放攝像頭資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()
