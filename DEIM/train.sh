export CUDA_VISIBLE_DEVICES=4,5,6
# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1
torchrun --master_port=7777 --nproc_per_node=3 train.py \
    -c /data/hyt/DEIM/configs/deim_rtdetrv2/deim_r101vd_60e_coco.yml \
    -t /data/hyt/DEIM/deim_rtdetrv2_r101vd_coco_60e.pth \
    --seed=0