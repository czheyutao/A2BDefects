export CUDA_VISIBLE_DEVICES=5
python tools/analysis_tools/benchmark.py \
    '/data/hyt/mmdetection/myconfig/mask2former_r50_8xb2-lsj-50e_coco.py' \
    --checkpoint '/data/hyt/mmdetection/mask2former/iter_20000.pth' \
    --fuse-conv-bn