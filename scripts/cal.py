import os
import json
import glob

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def average_metrics_in_subdirectories(root_dir, prefix):
    metrics_sum = {}
    metrics_count = {}

    # 遍历根目录下的所有子目录
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        # 确保当前子目录是目录且以给定前缀开头
        if os.path.isdir(subdir_path):
            json_file_path = os.path.join(subdir_path, 'eval', 'metrics', f'{prefix}_*.json')

            # 获取当前子目录下所有以给定前缀开头的json文件路径
            json_files = glob.glob(json_file_path)

            # 遍历每个json文件，读取其中的metrics
            for json_file in json_files:
                metrics_data = read_json_file(json_file)
                for key, value in metrics_data.items():
                    if key not in metrics_sum:
                        metrics_sum[key] = 0.0
                        metrics_count[key] = 0
                    metrics_sum[key] += value
                    metrics_count[key] += 1

    # 计算平均值
    metrics_avg = {key: metrics_sum[key] / metrics_count[key] for key in metrics_sum}

    # 保存平均值到txt文件
    with open(f'{root_directory}/average_metrics.txt', 'a') as f:
        f.write(f'{root_directory}\n')
        for key, value in metrics_avg.items():
            f.write(f'{key}: {value}\n')
        f.write(f'\n')

# 指定目录和文件前缀
root_directory = '/data1/hn/gaussianSim/gs4d/gs_1/work_dirs/0511/phase1/dynamic/nvs50'
file_prefix = '50000_images_full'

# 计算并保存平均值
average_metrics_in_subdirectories(root_directory, file_prefix)
