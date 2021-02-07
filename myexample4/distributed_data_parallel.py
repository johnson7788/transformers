#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/2/5 3:55 下午
# @File  : t1.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 分布式训练测试脚本, 只有node_rank是不同的
# 节点1上运行:
# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="192.168.50.139" --master_port=1234 distributed_data_parallel.py
# 节点2上运行:
# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="192.168.50.139" --master_port=1234 distributed_data_parallel.py

# 单机测试:
# python distributed_data_parallel.py

import torch
import argparse
import os
# FOR DISTRIBUTED: (can also use torch.nn.parallel.DistributedDataParallel instead)
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

# FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
# the 'WORLD_SIZE' environment variable will also be set automatically.
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

# Each process receives its own batch of "fake input data" and "fake target data."
# The "training loop" in each process just uses this fake batch over and over.
# https://github.com/NVIDIA/apex/tree/master/examples/imagenet provides a more realistic
# example of distributed data sampling for both training and validation.
x = torch.randn(N, D_in, device='cuda')
y = torch.randn(N, D_out, device='cuda')

model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

if args.distributed:
    # FOR DISTRIBUTED:  After amp.initialize, wrap the model with
    # apex.parallel.DistributedDataParallel.
    model = DistributedDataParallel(model)
    # torch.nn.parallel.DistributedDataParallel is also fine, with some added args:
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank)

loss_fn = torch.nn.MSELoss()

for t in range(500):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

if args.local_rank == 0:
    print("final loss = ", loss)
