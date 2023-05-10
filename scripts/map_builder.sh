#!/bin/bash

cd ~/repos/text2task
PYTHONPATH=$PWD/src python3 src/tools/map_builder.py config/start.json
