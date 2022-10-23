#!/bin/bash

cd ~/repos/text2task/src
python tools/produce_datasets.py

cd ~/repos/text2task
rm -rf work_dir/*

    # --validation_file ../training_data_tmp/val.txt \

cd ~/repos/text2task/work_dir
python ../src/tools/train.py \
    --model_name_or_path distilgpt2 \
    --train_file ../training_data_tmp/train.txt \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --output_dir /mnt/e/work_dirs/text2task_distilgpt2/ \
    --overwrite_output_dir \
    --overwrite_cache
