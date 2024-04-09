from moviepy.editor import VideoFileClip
import os

input_folder_path = 'input'

# 列出所有非mp4檔案
files = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f)) and not f.endswith('.mp4')]

for file in files:
    original_file_path = os.path.join(input_folder_path, file)
    output_file_path = os.path.join(input_folder_path, os.path.splitext(file)[0] + '.mp4')
    
    # 使用moviepy讀取影片
    clip = VideoFileClip(original_file_path)
    
    # 獲取原始影片的寬度和高度
    original_width = clip.size[0]
    original_height = clip.size[1]
    
    # 檢查是否需要旋轉影片（如果原始影片是垂直的）
    if original_width < original_height:
        # 如果影片已經是垂直的，按原始比例交換寬高
        new_width = original_height
        new_height = original_width
    else:
        # 如果影片是水平的，也按原始比例交換寬高（這部分根據你的具體需求可能需要調整）
        new_width = original_height
        new_height = original_width

    # 調整影片尺寸
    resized_clip = clip.resize(newsize=(new_width, new_height))
    
    # 保存為mp4格式
    resized_clip.write_videofile(output_file_path)

    # 刪除原始檔案（如果需要）
    # os.remove(original_file_path)
