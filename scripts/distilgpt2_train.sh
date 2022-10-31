#!/bin/bash

cd ~/repos/text2task
rm -rf work_dir/*

cd ~/repos/text2task/work_dir
python ../src/tools/train.py ../config/test_finetune.json
