#!/bin/bash

for number in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
#for number in 0 1
do
python ~/workspace/gridtools/gridtools-master/gridtools/qsub.py --mem 62000 --gpus 1 --slots 2 --local False --maxtime "'162:0:0'" ./train.py --isDebug=False --num_steps=1000 --generate_every=5 --wavenet_params=./params/$number/wavenet_params.json --logdir_root=./params/$number --data_dir=./params/$number --batch_size=3 &
#python3 train.py --isDebug=True --num_steps=1 --generate_every=2 --wavenet_params=./params/$number/wavenet_params.json --logdir_root=./params/$number --data_dir=./params/$number &
done
exit 0
