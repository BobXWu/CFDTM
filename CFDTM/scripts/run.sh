#!/bin/bash
set -e
dataset=${1}
K=${2:-50}
index=${3:-1}

python dynamic_topic_model.py --model_config CFDTM --dataset ${dataset} --num_topic ${K}

python utils/eva.py --data_dir ../data/${dataset}/ --topic_path output/${dataset}/CFDTM_K50_${index}th_T15
