#!/bin/bash

cd ~/repos/text2task
rm -rf work_dir/*

# rm -rf /mnt/e/work_dirs/text2task_distilgpt2/*

cd ~/repos/text2task/work_dir
PYTHONPATH=$PWD/../src python3 ../src/tools/train.py ../config/start.json

