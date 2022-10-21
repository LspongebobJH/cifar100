#!/bin/bash

python=/home/ubuntu/optim/.venv/bin/python
if [ $1 == 'ws' ]; then
    nohup $python train.py --ft-type ws > ./logs/logws.out 2>&1 &
elif [ $1 == 'all' ]; then
    nohup $python train.py --base > ./logs/logbase.out 2>&1 &
    nohup $python train.py --ft-type ws --batch-size 32 --device cuda:1 > ./logs/logws.out 2>&1 &
    nohup $python train.py --ft-type attn --batch-size 32 --device cuda:2 > ./logs/logattn.out 2>&1 &
fi

