#!/bin/bash

ntrain=32768
nval=4096
ntest=4096
ng=144
datapath='/path/to/data/poisson'

e1=1  # poissons diffusion eigenvalue range
e2=5

adr1=0.2 # advection to diffusion ratio range
adr2=1
# for AD ratio we saved a set of velocity scales that correspond to AD ration in utils/*.npy. See python script for details

o1=1 # helmholtz wave number range
o2=10

# create poissons examples
python utils/gen_data_poisson.py --ntrain=8192 --nval=4 --ntest=4 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/po/1-5-8192 --e1 1 --e2 5

python utils/gen_data_poisson.py --ntrain=40960 --nval=4096 --ntest=4096 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/po/5-7-40960 --e1 5 --e2 7


# create AD examples
python utils/gen_data_advdiff.py --ntrain=512 --nval=4 --ntest=4 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/ad/fine/ad2-4/512 --adr1 2 --adr2 4

python utils/gen_data_advdiff.py --ntrain=512 --nval=4 --ntest=4 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/ad/fine/ad4-6/512 --adr1 4 --adr2 6

python utils/gen_data_advdiff.py --ntrain=8192 --nval=4 --ntest=4 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/ad/2-4-8192 --adr1 2 --adr2 4

python utils/gen_data_advdiff.py --ntrain=16384 --nval=4 --ntest=4 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/ad/2-4-16384 --adr1 2 --adr2 4





python utils/gen_data_poisson.py --ntrain=2048 --nval=4 --ntest=4 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/po/fine --e1 5 --e2 7

# create Helm examples
python utils/gen_data_helmholtz.py --ntrain=8192 --nval=4096 --ntest=4096 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/helm/helm1-5-8192 --o1 1 --o2 5

python utils/gen_data_helmholtz.py --ntrain=2048 --nval=4 --ntest=4 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/helm/fine --o1 1 --o2 7


#
python utils/gen_data_advdiff.py --ntrain=40960 --nval=4096 --ntest=4096 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/ad/0.2-2-40960 --adr1 0.2 --adr2 2

#0604 一堆数据集
python utils/gen_data_advdiff.py --ntrain=2048 --nval=4 --ntest=4 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/ad/0.2-2-2048 --adr1 0.2 --adr2 2

#0605 main3
python utils/gen_data_advdiff.py --ntrain=32768 --nval=4 --ntest=4 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/ad/2-4-32768 --adr1 2 --adr2 4



# 方法里要补的图
python utils/gen_data_advdiff.py --ntrain=4 --nval=4 --ntest=4096 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/ad/5-6-test --adr1 5 --adr2 6

python utils/gen_data_poisson.py --ntrain=4 --nval=4 --ntest=4096 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/po/1-5-test --e1 1 --e2 5

python utils/gen_data_helmholtz.py --ntrain=4 --nval=4 --ntest=4096 --ng=144 --sparse --n 128 --datapath /root/autodl-tmp/data/helm/helm1-5-8192 --o1 1 --o2 5