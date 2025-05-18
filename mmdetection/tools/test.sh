export CUDA_VISIBLE_DEVICES=6,7
CONFIG_FILE=/data/hyt/mmdetection/myconfig/mask2former_r50_8xb2-lsj-50e_coco.py
CHECKPOINT_FILE=/data/hyt/mmdetection/mask2former/iter_20000.pth
GPU_NUM=2
./tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \