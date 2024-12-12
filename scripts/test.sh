GPU_NUM=1
TEST_CONFIG_YAML="configs/modl.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python test_modl.py \
    --config=$TEST_CONFIG_YAML
