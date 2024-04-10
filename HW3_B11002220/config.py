'''
Author: Leo lion24161582@gmail.com
Date: 2024-04-09 23:58:27
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-04-10 23:23:24
FilePath: \HW3_B11002220\config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# main.py 及時辨識參數
MODEL_PATH="weights\\best_120_v7.pt"

# videoTest.py參數 用來快速產生結果影片 
TEST_VIDEO_FOLDER_PATH = 'input'
OUTPUT_FOLDER_PATH = 'old_weights_output'
WEIGHT_FOLDER_PATH = 'old_weights'
VISUALIZE = True 
# Visualize.py參數
VISUALIZE_DIR = 'old_weights_output'

# Train.py參數
# TRAIN_PATH 訓練的資料夾 
TRAIN_PATH="Face_New-6"
EPOCHS = 240  # 訓練次數次數
SAVE_PERIOD = 80  # 每80 個epoch保存一次模型
