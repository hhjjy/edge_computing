import os
from collections import defaultdict

def find_and_group_files(output_dir):
    grouped_files = defaultdict(list)
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.txt'):
                # 使用文件名（例如 1_result.txt）作为键进行分组
                grouped_files[file].append(os.path.join(root, file))
    return grouped_files
def parse_file(filepath):
    data = {}
    with open(filepath, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            data[key] = int(value)
    return data
import matplotlib.pyplot as plt

def plot_data(grouped_files):
    for filename, paths in grouped_files.items():
        counts = {'hua': [], 'leo': []}
        labels = []
        for path in paths:
            data = parse_file(path)
            counts['hua'].append(data.get('hua', 0))
            counts['leo'].append(data.get('leo', 0))
            # 只使用权重名和视频编号作为标签
            label = path.split('\\')[-2] + '\n' + path.split('\\')[-1].replace('.txt', '')
            labels.append(label)

        x = range(len(paths))
        plt.figure(figsize=(10, 6))  # 调整图表尺寸
        width = 0.35  # 柱状图的宽度
        plt.bar(x, counts['hua'], width, label='hua', color='blue', alpha=0.6)
        plt.bar(x, [c+h for c,h in zip(counts['leo'], counts['hua'])], width, label='leo', color='red', alpha=0.6, bottom=counts['hua'])
        
        plt.xlabel('Weights and Video Number')
        plt.ylabel('Counts')
        plt.title(f'Comparison of {filename} across different weights')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        plt.show()


output_dir = 'output'
grouped_files = find_and_group_files(output_dir)
plot_data(grouped_files)
