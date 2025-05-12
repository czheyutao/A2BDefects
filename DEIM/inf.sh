export CUDA_VISIBLE_DEVICES=7
# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1
python '/data/hyt/DEIM/tools/inference/torch_inf.py'  \
    -c configs/deim_dfine/deim_hgnetv2_l_coco.yml \
    -r /data/hyt/DEIM/outputs/deim_hgnetv2_l_coco/best_stg1.pth \
    -d cuda:0 \
    --images_dir /data/hyt/mmdetection/WZ/test \
    -i /data/hyt/mmdetection/WZ/annotations/instances_test.json \