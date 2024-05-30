#!/bin/bash
data_list=(
    "/data1/hn/gaussianSim/data/processed/dynamic32/training/016"
    "/data1/hn/gaussianSim/data/processed/dynamic32/training/021"
    "/data1/hn/gaussianSim/data/processed/dynamic32/training/022"
)


DATE=$(date '+%m%d')
output_root="./work_dirs/$DATE/phase1/dynamic"
project=recon50

for data_dir in "${data_list[@]}"; do
    # 获取子目录的basename
    model_name=$(basename "$data_dir")


    # 使用basename来修改model_path
    model_path="$output_root/$project/$model_name"

    # 执行相同的命令，只修改-s和--model_path参数
    CUDA_VISIBLE_DEVICES=$1 proxychains python train.py \
        -s "$data_dir" \
        --model_path "$model_path" \
        --expname 'waymo' \

done
# bash scripts/dynamic/run_recon_list_d.sh 2