# encode=utf-8

import logging

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def load_train(cfg):
    logger = logging.getLogger('main.load')
    # 获取数据集，预处理
    logger.info('Preparing training data.')
    # 图像预处理
    transform_train = transforms.Compose([
        transforms.Resize(cfg.MODEL.BACKBONE.INPUT_RESOLUTION),
        transforms.RandomCrop(cfg.MODEL.BACKBONE.INPUT_RESOLUTION, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=cfg.DATASETS.TRAIN[0],
        train=True,
        download=True,
        transform=transform_train
    )

    # 训练策略
    strategy = cfg.SOLVER.STRATEGY
    assert strategy == 'MBGD' or strategy == 'BGD' or strategy == 'SGD', \
        "Solver's strategy must in (MBGD, BGD, SGD), please check your configure file."
    if strategy == 'MBGD':
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.DATASETS.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
    elif strategy == 'BGD':
        train_loader = DataLoader(
            train_set,
            batch_size=len(train_set),
            num_workers=4,
        )
    else:  # 'SGD'
        train_loader = DataLoader(
            train_set,
            shuffle=True,
        )
    return train_loader


def load_test(cfg):
    logger = logging.getLogger('main.load')
    logger.info('Preparing testing data.')

    transform_test = transforms.Compose([
        transforms.Resize(cfg.MODEL.BACKBONE.INPUT_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10(
        root=cfg.DATASETS.VAL[0],
        train=False,
        download=True,
        transform=transform_test
    )
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    return test_loader
