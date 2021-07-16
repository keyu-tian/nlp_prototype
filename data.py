import random
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import DistributedSampler

train_file = '../../train.xlsx'
test_file = '../../test.xlsx'
save_file = '../../结果.xlsx'


def pre_process(title: str):
    t = title.replace('\xa0', ' ')
    t = t.replace('#', '')
    t = t.replace('&', '')
    t = t.replace('!', '！')
    t = t.replace('?', '？')
    t = t.replace(',', '，')
    t = t.replace(':', '：')
    t = t.replace(';', '；')
    t = t.replace('|', ' ')
    t = t.lower()
    for i in range(21):
        t = t.replace(f'({i})', '').replace(f'（{i}）', '')
    t = ''.join([x.strip() for x in t.split() if len(x.strip()) > 0]).strip()
    return t


CLS_KEYS = ['其他旅游', '其他医疗', '其他文艺', '体育', '军事', '娱乐', '房产', '教育', '汽车', '游戏', '科技', '财经']
NUM_CLASSES = len(CLS_KEYS)


def labels_to_strs(labels) -> List[str]:  # todo: 去掉torch
    if isinstance(labels, int):
        l = [labels]
    elif isinstance(labels, tuple):
        l = list(labels)
    elif not isinstance(labels, list):
        l = labels.tolist()
    else:
        l = labels
    
    while True:
        if not isinstance(l[0], list):
            indices = l
            break
        l = l[0]
    
    labels = []
    for i in indices:
        if CLS_KEYS[i].startswith('其他'):
            labels.append('其他')
        else:
            labels.append(CLS_KEYS[i])
    return labels


def read_train_xlsx(is_tuning_hyper_parameters: bool):
    full_train = not is_tuning_hyper_parameters
    all_df = pd.read_excel(train_file, engine='openpyxl', sheet_name=None, header=None)
    keys = sorted(list(all_df.keys()))
    keys[0], keys[3] = keys[3], keys[0]
    assert keys == CLS_KEYS
    
    all_texts, all_labels = [], []
    tr_texts, tr_labels = [], []
    va_texts, va_labels = [], []
    
    random.seed(0)
    for label, k in enumerate(keys):
        ts = [pre_process(str(t[0])) for t in all_df[k].values]
        ts = list(filter(lambda x: 0 < len(x) <= 80, ts))
        for _ in range(3):
            random.shuffle(ts)
        
        tr_size = round(len(ts) * 0.9)
        # print(f'{len(ts)} vs {len(ts)-tr_size}')
        all_texts.extend(ts), all_labels.extend([label] * len(ts))
        tr_texts.extend(ts[:tr_size]), tr_labels.extend([label] * tr_size)
        va_texts.extend(ts[tr_size:]), va_labels.extend([label] * (len(ts) - tr_size))
    
    return (
        (all_texts, all_labels, va_texts, va_labels) if full_train else
        (tr_texts, tr_labels, va_texts, va_labels)
    )


def read_test_xlsx():
    test_df = pd.read_excel(test_file, engine='openpyxl')
    test_texts = [pre_process(s) for s in test_df['title']]
    return test_texts


def save_test_xlsx(labels):
    assert len(labels) == 12735
    test_df: pd.DataFrame = pd.read_excel(test_file, engine='openpyxl')
    test_df['channelName'] = labels_to_strs(labels)
    test_df.to_excel(save_file, engine='openpyxl', sheet_name='类别', index=False)


def __get_inp_mask_tar(tokenizer, texts: List[str], labels: List[int] = None):
    import torch
    data_dict = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
    input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']
    
    if labels is None:
        labels = torch.zeros(input_ids.shape[0], dtype=torch.long)
    else:
        labels = torch.tensor(labels, dtype=torch.long)
    labels = labels.reshape(-1).long()
    
    return input_ids, attention_mask, labels


def get_dataloader(dist, texts, labels, tokenizer, train, bs):
    from torch.utils.data import TensorDataset, DataLoader
    
    inp, msk, tar = __get_inp_mask_tar(tokenizer, texts, labels)
    dataset = TensorDataset(inp, msk, tar)
    if train:
        sp = DistributedSampler(dataset, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=bs, sampler=sp, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    else:
        sp, loader = None, DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    return sp, loader


if __name__ == '__main__':
    train_t, train_l, val_t, val_l = read_train_xlsx(is_tuning_hyper_parameters=False)
    # is  tuning: len(tt)=39225, tt[-1]=年内a股ipo、再融资规模近1.5万亿元新兴企业加速登陆资本市场,    len(vt)=4358, vt[-1]=上半年新增信贷已完成全年目标60%制造业中小微仍是下半年支持重点
    # not tuning: len(tt)=43583, tt[-1]=上半年新增信贷已完成全年目标60%制造业中小微仍是下半年支持重点, len(vt)=4358, vt[-1]=上半年新增信贷已完成全年目标60%制造业中小微仍是下半年支持重点
    print(f'len(tt)={len(train_t)}, tt[-1]={train_t[-1]}, len(vt)={len(val_t)}, vt[-1]={val_t[-1]}')
    
    test_t = read_test_xlsx()
    # print(type(ls), type(ls[0]), len(ls), ls[0], ls[-1], sep='\n')
    
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained("luhua/chinese_pretrain_mrc_macbert_large")
    
    train_dict = tokenizer(train_t, padding=True, truncation=False, return_tensors="pt")
    test_dict = tokenizer(test_t, padding=True, truncation=False, return_tensors="pt")
    
    train_input_ids, test_input_ids = train_dict['input_ids'], test_dict['input_ids']
    train_masks, test_masks = train_dict['attention_mask'], test_dict['attention_mask']
    
    print(train_input_ids[0].shape)
    print(test_input_ids[0].shape)
