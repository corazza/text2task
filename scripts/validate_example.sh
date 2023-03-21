#!/bin/bash

cd ~/repos/text2task
PYTHONPATH=$PWD/src python3 src/tools/validate_example.py config/start.json
