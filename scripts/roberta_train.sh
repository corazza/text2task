#!/bin/bash

cd ~/repos/text2task/src
python tools/produce_datasets.py

cd ~/repos/text2task/work_dir
python ../src/tools/run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file ../training_data_tmp/train.txt \
    --validation_file ../training_data_tmp/val.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --line_by_line \
    --output_dir /mnt/e/work_dirs/text2task_roberta/
