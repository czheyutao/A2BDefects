export CUDA_VISIBLE_DEVICES=5
python tools/analysis_tools/benchmark.py \
    '/data/hyt/Co-DETR-main/codino/co_dino_5scale_vit_large_coco_instance_batch1.py' \
    '/data/hyt/Co-DETR-main/codino/epoch_36.pth' \
    --fuse-conv-bn