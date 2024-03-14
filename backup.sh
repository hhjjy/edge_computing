#!/bin/bash

# 远程 Jetson 用户名和主机
remote_user="jetson"
remote_host="172.20.10.4"

# 远程截图目录的路径
remote_screenshot_dir="/home/$remote_user/screenshot"

# 本地备份目录的路径
local_backup_dir="./backup"
local_program_dir="$local_backup_dir/program"
local_screenshot_dir="$local_backup_dir/screenshot"

# 日志文件路径
log_file="./backup_log.txt"

# 确保本地备份目录和子目录存在
mkdir -p "$local_program_dir" "$local_screenshot_dir"

# 检查是否有参数传入
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <远程程序目录1> [<远程程序目录2> ...]"
    exit 1
fi

# 备份远程截图目录
echo "正在備份截圖目錄..." | tee -a "$log_file"
if ! rsync -avz "$remote_user@$remote_host:$remote_screenshot_dir/" "$local_screenshot_dir" >> "$log_file" 2>&1; then
    echo "备份截图目录失败，请检查日志：$log_file" | tee -a "$log_file"
    exit 1
fi

# 遍历所有传入的远程程序目录参数
for remote_program_dir in "$@"; do
    program_name=$(basename "$remote_program_dir")
    echo "正在備份目錄：$remote_program_dir..." | tee -a "$log_file"
    if ! rsync -avz "$remote_user@$remote_host:$remote_program_dir/" "$local_program_dir/$program_name" >> "$log_file" 2>&1; then
        echo "备份程序目录 $remote_program_dir 失败，请检查日志：$log_file" | tee -a "$log_file"
        # 考虑是否在此处退出或继续尝试备份其他目录
    fi
done

echo "備份完成" | tee -a "$log_file"
