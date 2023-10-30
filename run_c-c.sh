#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u train.py --os_dataset cifar100 --partial_rate 0.1 --cs_rate 0.2 --os_rate 0.4 --warmup 30 --alpha 1 --beta 0.1 --seed 5406

CUDA_VISIBLE_DEVICES=0 python -u train.py --os_dataset cifar100 --partial_rate 0.1 --cs_rate 0.3 --os_rate 0.6 --warmup 30 --alpha 1 --beta 0.1 --seed 5406

CUDA_VISIBLE_DEVICES=1 python -u train.py --os_dataset cifar100 --partial_rate 0.3 --cs_rate 0.2 --os_rate 0.4 --warmup 30 --alpha 1 --beta 0.1 --seed 5406

CUDA_VISIBLE_DEVICES=1 python -u train.py --os_dataset cifar100 --partial_rate 0.3 --cs_rate 0.3 --os_rate 0.6 --warmup 30 --alpha 1 --beta 0.1 --seed 5406

CUDA_VISIBLE_DEVICES=2 python -u train.py --os_dataset cifar100 --partial_rate 0.5 --cs_rate 0.2 --os_rate 0.4 --warmup 50 --alpha 1 --beta 0.1 --seed 5406

CUDA_VISIBLE_DEVICES=2 python -u train.py --os_dataset cifar100 --partial_rate 0.5 --cs_rate 0.3 --os_rate 0.6 --warmup 50 --alpha 1 --beta 0.1 --seed 5406
