# encode=utf-8
import argparse
import logging
import os

import torch
from torch import cuda
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import *

from models.imageclassifier import ImageClassifier
from solver.data_loader import load_train, load_test
from solver.optimizer_maker import make_optimizer, make_lr_scheduler
from utils.collect_env import collect_env_info
from utils.config_loading import load_config
from utils.utils import progress_bar

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk')


# 训练阶段
def do_train(epoch, model, optimizer, train_loader, test_loader, cfg):
    
    # 定义度量和优化
    criterion = CrossEntropyLoss()
    # switch to train mode
    model.train()
    logger = logging.getLogger('train')
    train_loss = 0
    train_correct = 0
    total = 0
    # batch 数据
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 数据移到GPU上
        if cfg.SOLVER.CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()
        # 先将optimizer梯度置为0
        optimizer.zero_grad()
        # Variable 表示该变量属于计算图的一部分，此处是图计算的开始处。图的leaf variable
        inputs, targets = Variable(inputs), Variable(targets)
        # 模型输出
        outputs = model(inputs)
        # 计算loss，图的终点处
        loss = criterion(outputs, targets)
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # loss不能直接相加，要用loss.data。loss是计算图的一部分，如果直接加loss，代表total loss同样属于模型一部分，那么图就越来越大
        train_loss += loss.data
        # 数据统计
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        train_correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx >= 1:
            break

    # 切到测试模型
    model.eval()
    test_loss = 0
    test_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if cfg.SOLVER.CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        test_correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx >= 1:
            break

    if (epoch + 1) % cfg.SOLVER.CHECKPOINT == 0:
        save(model, cfg)

    state = {
        'lr': '%.5f' % optimizer.state_dict()['param_groups'][0]['lr'],
        'loss': '%.3f (%.3f)'
                % (float(train_loss / len(train_loader)),
                   float(test_loss / len(test_loader))),
        'acc': '%.3f%% (%.3f%%)'
               % (float(train_correct) / len(train_loader.dataset.targets) * 100,
                  float(test_correct) / len(test_loader.dataset.targets) * 100),
    }
    progress_bar(logger, epoch + 1, cfg.SOLVER.MAX_ITER, state)
    model.epoch += 1


def train(model, train_loader, test_loader, cfg):

    current_iter = model.epoch
    max_iter = cfg.SOLVER.MAX_ITER

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    for epoch in range(current_iter, max_iter):
        scheduler.step(epoch)
        do_train(epoch, model, optimizer, train_loader, test_loader, cfg)
        # 清除部分无用变量
        cuda.empty_cache()
    save(model, cfg)
        

def save(model, cfg):
    logger = logging.getLogger('train.save')
    logger.info('Saving model')
    if not os.path.isdir(cfg.SOLVER.OUTPUT_DIR):
        os.mkdir(cfg.SOLVER.OUTPUT_DIR)
    name_len = len(str(cfg.SOLVER.MAX_ITER))
    model_name = str(model.epoch).zfill(name_len)
    torch.save(model, cfg.SOLVER.OUTPUT_DIR + model_name + '.pth')
    with open(cfg.SOLVER.OUTPUT_DIR + 'checkpoint', 'w') as f:
        f.write(model_name)
    logger.info('Model saved')

        
def main():
    # init logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger('main')

    # 获取参数
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--config-file', default='./config/ResNet-50.yaml', type=str,
                        help='configures loaded from this file')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    # Load checkpoint
    if os.path.isdir('checkpoint'):
        logging.info('Resuming from checkpoint.')
        f = open('./checkpoint/checkpoint')
        model_name = f.readline()
        model = torch.load('./checkpoint/%s.pth' % model_name)
        model.eval()
        # 如果模型存在，cfg就从模型中获得。否则会导致配置参数不一致
        cfg = model.cfg
        cfg.freeze()
    else:
        logger.info('Building model')
        cfg = load_config(args.config_file)
        use_cuda = cuda.is_available()

        # 检测 CUDA 配置和系统是否匹配
        if not use_cuda and cfg.SOLVER.CUDA:
            logger.error("The device doesn't support CUDA.")
            assert "The device doesn't support CUDA. Please check the configure file."
        elif use_cuda and not cfg.SOLVER.CUDA:
            logger.warning("This device supports CUDA, but disabled by config file.")

        cfg.freeze()
        model = ImageClassifier(cfg)

    if cfg.SOLVER.CUDA:
        model.cuda()
        # speed up slightly
        cudnn.benchmark = True

    logger.info('\n', '\n'.join(model.get_model_state()))

    train_loader = load_train(cfg)
    test_loader = load_test(cfg)

    logger.info('Start training')
    train(model, train_loader, test_loader, cfg)


if __name__ == '__main__':
    main()
