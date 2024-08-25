#!/usr/bin/bash

# # 单机单卡直接运行
python train.py \
    --train_dirs data/baidu \
                 data/skypile \
    --eval_files data/dev10K_2020-40_zh_head_0009.parquet \
    --model_init_path Qwen1.5-0.5B \
    --model_save_dir model_save \
    --iterable_train_dataset \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --report_to wandb \
    --save_steps 1000 \
    --eval_steps 2000 \
    --warmup_ratio 0.02 \
    --log_steps 5 \
    --save_total_limit 2 \
    | tee -a pre_train_log.txt


# 单机多卡使用 accelerate 运行
# accelerate launch --multi_gpu  --config_file accelerate_multi_gpu.yaml \
#     train.py \
#     --train_files ../autodl-tmp/data/wiki_chinese/wiki_fi.parquet \
#                   ../autodl-tmp/data/Baidu/baike_chunk_512_5.6M_0.parquet \
#                   ../autodl-tmp/data/Baidu/baike_chunk_512_5.6M_1.parquet \
#     --per_device_train_batch_size 8 \
#     --eval_accumulation_steps 8 \
#     --warmup_steps  10000\
#     --model_init_path ../autodl-tmp/Qwen1.5-0.5B \
#     --model_save_dir ../autodl-tmp/model_save/pre \
#     --report_to wandb \
#     --save_steps 5000 \
#     --log_steps 10 \
#     --save_total_limit 1 \
#     --iterable_train_dataset \
#     | tee -a pre_train_log_multi_gpu.txt
