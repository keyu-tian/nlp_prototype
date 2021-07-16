import datetime
import heapq
import math
import os
import time
from collections import defaultdict

import torch
from torch import nn as nn


def time_str():
    return datetime.datetime.now().strftime('[%m-%d %H:%M:%S]')


def master_echo(is_master, msg: str, color='33', tail=''):
    if is_master:
        os.system(f'echo -e "\033[{color}m{msg}\033[0m{tail}"')


def get_bn(bn_mom):
    def BN_func(*args, **kwargs):
        kwargs.update({'momentum': bn_mom})
        return nn.BatchNorm2d(*args, **kwargs)
    
    return BN_func


class TopKHeap(list):
    
    def __init__(self, maxsize):
        super(TopKHeap, self).__init__()
        self.maxsize = maxsize
        assert self.maxsize >= 1
    
    def push_q(self, x):
        if len(self) < self.maxsize:
            heapq.heappush(self, x)
        elif x > self[0]:
            heapq.heappushpop(self, x)
    
    def pop_q(self):
        return heapq.heappop(self)
    
    def __repr__(self):
        return str(sorted([x for x in self], reverse=True))


class AverageMeter(object):
    def __init__(self, length=0):
        self.length = round(length)
        if self.length > 0:
            self.queuing = True
            self.val_history = []
            self.num_history = []
        self.val_sum = 0.0
        self.num_sum = 0.0
        self.last = 0.0
        self.avg = 0.0
    
    def reset(self):
        if self.length > 0:
            self.val_history.clear()
            self.num_history.clear()
        self.val_sum = 0.0
        self.num_sum = 0.0
        self.last = 0.0
        self.avg = 0.0
    
    def update(self, val, num=1):
        self.val_sum += val * num
        self.num_sum += num
        self.last = val / num
        if self.queuing:
            self.val_history.append(val)
            self.num_history.append(num)
            if len(self.val_history) > self.length:
                self.val_sum -= self.val_history[0] * self.num_history[0]
                self.num_sum -= self.num_history[0]
                del self.val_history[0]
                del self.num_history[0]
        self.avg = self.val_sum / self.num_sum
    
    def time_preds(self, counts):
        remain_secs = counts * self.avg
        remain_time = datetime.timedelta(seconds=round(remain_secs))
        finish_time = time.strftime("%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
        return remain_time, finish_time


def adjust_learning_rate(optimizer, cur_iter, max_iter, max_lr):
    warmup_iters = max(max_iter // 100, 2)
    if cur_iter <= warmup_iters:
        ratio = cur_iter / warmup_iters
        base_lr = max_lr / 5
        cur_lr = base_lr + ratio * (max_lr - base_lr)
    else:
        ratio = (cur_iter - warmup_iters) / (max_iter - 1 - warmup_iters)
        cur_lr = max_lr * 0.5 * (1. + math.cos(math.pi * ratio))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def filter_params(model: torch.nn.Module):
    special_decay_rules = {
        'bn_b': {'weight_decay': 0.0},
        'bn_w': {'weight_decay': 0.0},
    }
    pgroup_normal = []
    pgroup = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    names = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    if 'conv_dw_w' in special_decay_rules:
        pgroup['conv_dw_w'] = []
        names['conv_dw_w'] = []
    if 'conv_dw_b' in special_decay_rules:
        pgroup['conv_dw_b'] = []
        names['conv_dw_b'] = []
    if 'conv_dense_w' in special_decay_rules:
        pgroup['conv_dense_w'] = []
        names['conv_dense_w'] = []
    if 'conv_dense_b' in special_decay_rules:
        pgroup['conv_dense_b'] = []
        names['conv_dense_b'] = []
    if 'linear_w' in special_decay_rules:
        pgroup['linear_w'] = []
        names['linear_w'] = []
    
    names_all = []
    type2num = defaultdict(lambda: 0)
    for name, m in model.named_modules():
        clz = m.__class__.__name__
        if clz.find('Conv2d') != -1:
            if m.bias is not None and m.bias.requires_grad:
                if 'conv_dw_b' in pgroup and m.groups == m.in_channels:
                    pgroup['conv_dw_b'].append(m.bias)
                    names_all.append(name + '.bias')
                    names['conv_dw_b'].append(name + '.bias')
                    type2num[m.__class__.__name__ + '.bias(dw)'] += 1
                elif 'conv_dense_b' in pgroup and m.groups == 1:
                    pgroup['conv_dense_b'].append(m.bias)
                    names_all.append(name + '.bias')
                    names['conv_dense_b'].append(name + '.bias')
                    type2num[m.__class__.__name__ + '.bias(dense)'] += 1
                else:
                    pgroup['conv_b'].append(m.bias)
                    names_all.append(name + '.bias')
                    names['conv_b'].append(name + '.bias')
                    type2num[m.__class__.__name__ + '.bias'] += 1
            if 'conv_dw_w' in pgroup and m.groups == m.in_channels and m.weight.requires_grad:
                pgroup['conv_dw_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['conv_dw_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight(dw)'] += 1
            elif 'conv_dense_w' in pgroup and m.groups == 1 and m.weight.requires_grad:
                pgroup['conv_dense_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['conv_dense_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight(dense)'] += 1
        
        elif clz.find('Linear') != -1:
            if m.bias is not None and m.bias.requires_grad:
                pgroup['linear_b'].append(m.bias)
                names_all.append(name + '.bias')
                names['linear_b'].append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
            if 'linear_w' in pgroup and m.weight.requires_grad:
                pgroup['linear_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['linear_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
        
        elif clz.find('BatchNorm2d') != -1:
            if m.weight is not None and m.weight.requires_grad:
                pgroup['bn_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['bn_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
            if m.bias is not None and m.bias.requires_grad:
                pgroup['bn_b'].append(m.bias)
                names_all.append(name + '.bias')
                names['bn_b'].append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
    
    for name, p in model.named_parameters():
        if name not in names_all and p.requires_grad:
            pgroup_normal.append(p)
    
    param_groups = [{'params': pgroup_normal}]
    for ptype in pgroup.keys():
        if ptype in special_decay_rules.keys():
            param_groups.append({'params': pgroup[ptype], **special_decay_rules[ptype]})
        else:
            param_groups.append({'params': pgroup[ptype]})
    
    return param_groups
