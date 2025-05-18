export CUDA_VISIBLE_DEVICES=5  # 限制只使用第5个GPU（编号为4）

# 定义训练参数
DATA_ROOT="/data/hyt/mmdetection/WZ"
MODEL_TYPE="vit_b"
CHECKPOINT_PATH="./wz_b/sam_vit_b.pth"
OUTPUT_DIR="./wz_b"

# 运行训练脚本
python /data/hyt/SAM/finetune.py \
    --data_root $DATA_ROOT \
    --model_type $MODEL_TYPE \
    --checkpoint_path $CHECKPOINT_PATH \
    --batch_size 1 \
    --image_size 1024 \
    --steps 3000 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --metrics_interval 100 \
    --output_dir $OUTPUT_DIR \