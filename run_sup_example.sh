#!/bin/bash
NUM_GPU=7
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6\

PORT_ID=$(expr $RANDOM + 1000)

export OMP_NUM_THREADS=8

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID main.py \
    --model_name_or_path bert-base-uncased \
    --output_dir result/my-sup-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --evaluation_strategy steps \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --eval_steps 5 \
    --logging_steps 50 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
