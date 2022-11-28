#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 \
python main.py \
    --model_name_or_path my-unsup-bert-base-uncased/checkpoint \
    --output_dir result/my-unsup-simcse-bert-base-uncased/final \
    --num_train_epochs 1 \
    --per_device_train_batch_size 14 \
    --learning_rate 3e-5 \
    --max_seq_length 512 \
    --evaluation_strategy steps \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --save_steps 50000 \
    --logging_steps 250 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
