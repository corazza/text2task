#!/bin/bash

cd ~/repos/text2task
rm -rf work_dir/*

cd ~/repos/text2task/work_dir
PYTHONPATH=$PWD/../src python3 ../src/tools/train.py ../config/test_finetune.json
