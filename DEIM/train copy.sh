export CUDA_VISIBLE_DEVICES=4,5,6,7
# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1
torchrun --master_port=7777 --nproc_per_node=4 train.py \
    -c configs/deim_dfine/deim_hgnetv2_l_coco.yml \
    -t /data/hyt/DEIM/deim_dfine_hgnetv2_l_coco_50e.pth \
    --seed=0