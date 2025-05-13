export CUDA_VISIBLE_DEVICES=4,5,6
# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1
torchrun --master_port=7777 --nproc_per_node=4 train.py \
    -c /data/hyt/DEIM/configs/deim_rtdetrv2/rtdetrv2_r101vd_6x_coco.yml \
    --seed=0 \
    --test-only \
    -r /data/hyt/DEIM/outputs/deim_rtdetrv2_r101vd_60e_coco/best_stg1.pth \