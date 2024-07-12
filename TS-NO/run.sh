#!/bin/bash

# this run script assumes slurm job scheduler, but similar run cmd can be used elsewhere
# for DDP (slurm vars setting)
export MASTER_ADDR=$(hostname)

# number of gpus
ngpu=4

# yaml file
config_file=./config/operators_poisson.yaml
# config name to run
config="poisson-scale-k1_5"
# sub run number
run_num="test"

# where to store results
scratch="/path/to/results/"

# run command
cmd="python train.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$scratch"

# AD的训练命令
# python train.py --yaml_config=config/operators_ad.yaml --config="ad-scale-adr0p2_1" --run_num="semi" --root_dir="/hy-tmp/model_para/AD/2-4-8192-semi/"




# source DDP vars first for data-parallel training (if not srun, just source and then run cmd; see pytorch docs for DDP)
srun -l -n $ngpu --cpus-per-task=10 --gpus-per-node $ngpu bash -c "source export_DDP_vars.sh && $cmd"

# for inference run the following commands to use eval.py (single gpu is sufficient, no logging by default)
# pass the model weights to use
#weights_for_inference=$scratch/expts/$config/$run_num/checkpoints/ckpt_best.tar
#cmd_inf="python eval.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$scratch --weights=$weights_for_inference"
#bash -c "$cmd_inf"


# 注意力加权
python train.py --yaml_config=config/operators_ad.yaml --config="ad-scale-adr0p2_1" --run_num="FNO-TSW-pretrain" --root_dir="/root/autodl-tmp/model_para/"

python train.py --yaml_config=config/operators_ad.yaml --config="ad-scale-adr0p2_1" --run_num="TS-fno-16384" --root_dir="/root/autodl-tmp/model_para/"


python train.py --yaml_config=config/operators_ad.yaml --config="ad-scale-adr0p2_1" --run_num="weight" --root_dir="/root/autodl-tmp/model_para/"

python train.py --yaml_config=config/operators_ad.yaml --config="ad-scale-adr0p2_1" --run_num="cos_weight" --root_dir="/root/autodl-tmp/model_para/"


python train.py --yaml_config=config/operators_ad.yaml --config="ad-scale-adr0p2_1" --run_num="time" --root_dir="/root/autodl-tmp/model_para/"