#!/bin/bash

cd ~/repos/text2task
PYTHONPATH=$PWD/src python3 src/tools/produce_datasets.py
