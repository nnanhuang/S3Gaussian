CUDA_VISIBLE_DEVICES=$1 python /data1/hn/gaussianSim/gs4d/gs_1/preprocess_main.py \
    --data_root "/data1/hn/gaussianSim/data/dynamic32" \
    --target_dir "/data1/hn/gaussianSim/data/processed/dynamic32" \
    --split "training" \
    --process_keys "images" "lidar" "calib" "pose" "dynamic_masks" \
    --workers "1" \
    --scene_ids 21
    # --split_file "/data1/hn/gaussianSim/gs4d/gs_1/data/waymo_splits/dynamic32.txt"

# bash scripts/proprocess.sh 1