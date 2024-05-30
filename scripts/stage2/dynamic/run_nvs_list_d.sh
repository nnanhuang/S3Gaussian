#!/bin/bash


data_list=(
    # "/data1/hn/gaussianSim/data/processed/dynamic32/training/016"
    # "/data1/hn/gaussianSim/data/processed/dynamic32/training/021"
    "/data1/hn/gaussianSim/data/processed/dynamic32/training/022"
)


DATE=$(date '+%m%d')
output_root="./work_dirs/0512/phase2/dynamic"
project=nvs50

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
        --configs "arguments/stage2_nvs.py" \
        --start_checkpoint "/data1/hn/gaussianSim/gs4d/gs_1/work_dirs/0512/phase2/dynamic/nvs50/022/chkpnt_fine_30000.pth"
        # --prior_checkpoint "/data1/hn/gaussianSim/gs4d/gs_1/work_dirs/0511/phase1/dynamic/nvs50/$model_name/chkpnt_fine_50000.pth"
done
# bash scripts/stage2/dynamic/run_nvs_list_d.sh 6