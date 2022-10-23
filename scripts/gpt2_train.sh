#!/bin/bash

cd ~/repos/text2task/work_dir
python ../src/tools/run_clm.py \
    --model_name_or_path gpt2-medium \
    --train_file ../training_data_tmp/train.txt \
    --validation_file ../training_data_tmp/val.txt \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir /mnt/e/work_dirs/text2task_gpt2/
