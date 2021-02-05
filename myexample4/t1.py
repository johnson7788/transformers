#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/2/5 3:55 下午
# @File  : t1.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 分布式训练测试脚本, 只有node_rank是不同的
# 节点1上运行:
# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="192.168.50.139" --master_port=1234 t1.py --num_epochs 1 --batch_size 4
# 节点2上运行:
# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="192.168.50.139" --master_port=1234 t1.py --num_epochs 1 --batch_size 4

# 单机测试:
# python  t1.py --num_epochs 1 --batch_size 4 --local_rank -1

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def evaluate(model, device, test_loader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def main():
    num_epochs_default = 2
    batch_size_default = 16  # 1024
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "saved_models"
    model_filename_default = "resnet_distributed.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.",
                        default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="模型文件的位置,如果继续训练，需要指定", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="是否继续训练从保存的 checkpoint.")
    parser.add_argument("--no_cuda", action="store_true", help="是否使用cuda")
    args = parser.parse_args()

    local_rank = args.local_rank
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    random_seed = args.random_seed
    model_dir = args.model_dir
    model_filename = args.model_filename
    resume = args.resume

    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Encapsulate the model on the GPU assigned to the current process
    model = torchvision.models.resnet18(pretrained=True)
    if local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # 初始化分布式的后端 the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend="gloo")
        torch.distributed.init_process_group(backend='nccl')

    model = model.to(device)
    if local_rank == -1 or args.no_cuda:
        ddp_model = model
    else:
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_set)
    else:
        # Restricts data loading to a subset of the dataset exclusive to the current process
        train_sampler = DistributedSampler(train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))

        # Save and evaluate model routinely
        if epoch % 10 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                torch.save(ddp_model.state_dict(), model_filepath)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        ddp_model.train()

        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
