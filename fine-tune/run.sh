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

# python eval.py --yaml_config=/hy-tmp/eval/config/eval_ad-supervise.yaml --config="ad-scale-adr0p2_1" --run_num="test" --root_dir=eval_result/AD/zeroshot/ --weights=/hy-tmp/model_para/AD-fine/supervise/32/expts/ad-scale-adr0p2_1/ft-32-200epochs/checkpoints/ckpt.tar

# python eval.py --yaml_config=/hy-tmp/eval/config/eval_ad.yaml --config="ad-scale-adr0p2_1" --run_num="test" --root_dir=eval_result/AD/zeroshot/ --weights=/hy-tmp/model_para/AD/2-4-8192-con-exp/expts/ad-scale-adr0p2_1/con/checkpoints/ckpt_best.tar

# python eval.py --yaml_config=/hy-tmp/eval/config/eval_ad.yaml --config="ad-scale-adr0p2_1" --run_num="test" --root_dir=eval_result/AD/zeroshot/ --weights=/hy-tmp/model_para/AD/2-4-8192-con/expts/ad-scale-adr0p2_1/con/checkpoints/ckpt_best.tar

# python eval.py --yaml_config=/hy-tmp/eval/config/eval_ad.yaml --config="ad-scale-adr0p2_1" --run_num="test" --root_dir=eval_result/AD/zeroshot/ --weights=/hy-tmp/model_para/AD/2-4-16384-semi-2scale/expts/ad-scale-adr0p2_1/semi/checkpoints/ckpt_best.tar


# fine-tune命令

#python train.py --yaml_config='config/st-ad2-4-T=5/8.yaml' --config="ad-scale-adr0p2_1" --run_num="st-ad2-4-T=5-8" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/st-ad2-4-T=5/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="st-ad2-4-T=5-2048" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/st-ad2-4-T=30/8.yaml' --config="ad-scale-adr0p2_1" --run_num="st-ad2-4-T=30-8" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/st-ad2-4-T=30/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="st-ad2-4-T=30-2048" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/st-ad2-4-semi-norm/8.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-semi-norm-8" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/st-ad2-4-semi-norm/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-semi-norm-2048" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/ad2-4-atten/8.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-atten-8" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/ad2-4-atten/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-atten-2048" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/ad2-4-para_atten/8.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-para_atten-8" --root_dir="/root/autodl-tmp/fine_model/"

python train.py --yaml_config='config/ad2-4-para_atten/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-para_atten-4e-4-batch128-2048" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/ad2-4-distance/8.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-distance-8" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/ad2-4-distance/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-distance-2048" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/ad2-4-base/8.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-base-8" --root_dir="/root/autodl-tmp/fine_model/"

#python train.py --yaml_config='config/ad2-4-base/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ad2-4-base-2048" --root_dir="/root/autodl-tmp/fine_model/"

python train.py --yaml_config='config/ad4-6-nodata/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ad4-6-nodata" --root_dir="/root/autodl-tmp/fine_model/"

python train.py --yaml_config='config/ad4-6-para/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ad4-6-para_atten" --root_dir="/root/autodl-tmp/fine_model/"


python train.py --yaml_config='config/po5-7/2048.yaml' --config="poisson-scale-k1_5" --run_num="po5-7-atten-2048" --root_dir="/root/autodl-tmp/fine_model/"

python train.py --yaml_config='config/helm1-7/2048.yaml' --config="helm-scale-o1_10" --run_num="helm1-7-atten-2048" --root_dir="/root/autodl-tmp/fine_model/"

python train.py --yaml_config='config/ad4-6-88-2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ad4-6-para_atten-88" --root_dir="/root/autodl-tmp/fine_model/"

python train.py --yaml_config='config/ad4-6-816-2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ad4-6-para_atten-816" --root_dir="/root/autodl-tmp/fine_model/"


python train.py --yaml_config='config/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="FNO-PT-2048" --root_dir="/root/autodl-tmp/fine_model/"

python train.py --yaml_config='config/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="FNO-TSW-2048-4" --root_dir="/root/autodl-tmp/fine_model/"


python train.py --yaml_config='config/ad2-4-atten/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="ts-fno-16384" --root_dir="/root/autodl-tmp/fine_model/"

python train.py --yaml_config='config/ad2-4-atten/2048.yaml' --config="ad-scale-adr0p2_1" --run_num="fno-loss" --root_dir="/root/autodl-tmp/fine_model/"