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

# source DDP vars first for data-parallel training (if not srun, just source and then run cmd; see pytorch docs for DDP)
srun -l -n $ngpu --cpus-per-task=10 --gpus-per-node $ngpu bash -c "source export_DDP_vars.sh && $cmd"

# for inference run the following commands to use eval.py (single gpu is sufficient, no logging by default)
# pass the model weights to use
#weights_for_inference=$scratch/expts/$config/$run_num/checkpoints/ckpt_best.tar
#cmd_inf="python eval.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$scratch --weights=$weights_for_inference"
#bash -c "$cmd_inf"


python train.py --yaml_config=config/operators_ad4-6.yaml --config="ad-scale-adr0p2_1" --run_num="ad4-6-teacher" --root_dir="/root/autodl-tmp/model_para/"

python eval.py --yaml_config=config/from/operators_ad.yaml --config="ad-scale-adr0p2_1" --run_num="test1" --root_dir="/root/autodl-tmp/model_test" --weights="/root/autodl-tmp/model_para/expts/ad-scale-adr0p2_1/ad4-6-2048/checkpoints/ckpt_best.tar"

python eval.py --yaml_config=config/operators_ad4-6.yaml --config="ad-scale-adr0p2_1" --run_num="test1" --root_dir="/root/autodl-tmp/model_test" --weights="/root/autodl-tmp/fine_model/expts/ad-scale-adr0p2_1/ad4-6-para_atten-816/checkpoints/ckpt_best.tar"



# 引言里的实验
python train.py --yaml_config=config/operators_ad-0.2-2.yaml --config="ad-scale-adr0p2_1" --run_num="ad0.2-2-self" --root_dir="/root/autodl-tmp/model_para/"




# po
python train.py --yaml_config=config/operators_poisson.yaml --config="poisson-scale-k1_5" --run_num="po5-7-teacher" --root_dir="/root/autodl-tmp/model_para/"

python eval.py --yaml_config=config/from/operators_poisson.yaml --config="poisson-scale-k1_5" --run_num="test1" --root_dir="/root/autodl-tmp/model_test" --weights="/root/autodl-tmp/model_para/expts/poisson-scale-k1_5/po5-7-2048/checkpoints/ckpt_best.tar"


python train.py --yaml_config=config/operators_helmholtz.yaml --config="helm-scale-o1_10" --run_num="helm1-7-teacher" --root_dir="/root/autodl-tmp/model_para/"



# main2实验
python train.py --yaml_config=config/operators_ad.yaml --config="ad-scale-adr0p2_1" --run_num="time" --root_dir="/root/autodl-tmp/model_para/"

