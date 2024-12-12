GPU_NUM=1
TRAIN_CONFIG_YAML="configs/modl.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python train_modl.py \
    --config=$TRAIN_CONFIG_YAML
