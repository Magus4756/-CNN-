# encode=utf-8

# Some helper functions for PyTorch

import time

import torch
from torch.nn import *

# _, term_width = os.popen('stty size').read().split()
term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65
last_time = time.time()
begin_time = last_time


def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset
    :param dataset:
    :return:
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==>Computing mean and std...')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    for m in net.modules():
        if isinstance(m, Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def progress_bar(logger, current: int, total: int, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    output_states = ['%d/%d' % (current, total),
                     'time: %s' % format_time(step_time),
                     'total: %s' % format_time(tot_time)]
    for key in msg:
        output_states.append('%s: %s' % (key, msg[key]))

    msg = ' | '.join(output_states)
    logger.info(msg)


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    f += '%02d' % hours + ':'
    i += 1
    f += '%02d' % minutes + ':'
    i += 1
    f += '%02d' % secondsf + '.'
    i += 1
    f += '%03d' % millis
    i += 1
    if f == '':
        f = '00:00:00.000'
    return f


