'''
Author: Leo lion24161582@gmail.com
Date: 2024-04-10 12:11:59
LastEditors: Leo lion24161582@gmail.com
LastEditTime: 2024-04-10 17:13:58
FilePath: \edge_computing\HW3_B11002220\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import subprocess
import os
import logging
from config import *

# model = os.path.join(TRAIN_PATH,"best.pt")
model = "yolov8n.pt"  
epochs = EPOCHS 
save_period = SAVE_PERIOD
base_command = f"yolo task=detect mode=train model={model} data=data.yaml epochs={epochs} imgsz=640 plots=True batch=32 save_period={save_period}"
print(base_command)
# 配置日誌
logging.basicConfig(level=logging.INFO, filename='training_log.log',
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# 紀錄當前工作目錄
logger.info(f"當前目錄在變更前: {os.getcwd()}")

# 設置訓練路徑
train_path = TRAIN_PATH

# 開始訓練
logger.info(f"開始訓練，訓練週期為 {epochs} 次，每 {save_period} 次保存一次，在 {train_path}...")
subprocess.run(base_command, shell=True, cwd=train_path)
logger.info("訓練完成。")

logger.info(f"變更後的當前目錄: {os.getcwd()}")
