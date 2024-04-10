'''
Author: Leo lion24161582@gmail.com
Date: 2024-04-09 23:58:27
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-04-10 13:58:57
FilePath: \HW3_B11002220\config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# videoTest.py參數

MODEL_PATH="weights\\best_120_v7.pt"
PROCESS_ALL_WEIGHTS = False

# Train.py參數
TRAIN_PATH="Face_New-3"
EPOCHS = 240  # 訓練次數次數
SAVE_PERIOD = 80  # 每80 個epoch保存一次模型
# Visualize.py參數
VISUALIZE = True 
