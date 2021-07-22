import random
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import DistributedSampler

train_file = '../../raw_train.xlsx'
test_file = '../../test.xlsx'
save_file = '../../result.xlsx'


def pre_process(content: str, title: str):
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

    c = content.replace('(', '（').replace(')', '）')
    c = c.replace('[', '（').replace(']', '）')
    c = c.replace('{', '（').replace('}', '）')
    
    ls, sta = [], []
    try:
        for ch in c:
            if ch == '（':
                sta.append('（')
            if len(sta) == 0 and ch not in {'\r', '\n', '\t'}:
                ls.append(ch)
            if ch == '）' and len(sta) > 0:
                sta.pop()
    except:
        print(c)
        exit(-1)
    c = ' '.join(''.join(ls).split())[:471-len(t)]
    return c, t


CLS_KEYS = ['体育', '其他健康', '其他宠物', '其他慈善', '其他政治', '其他文艺', '其他旅游', '其他育儿', '军事', '娱乐', '房产', '教育', '汽车', '游戏', '科技', '财经']
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
    assert keys == CLS_KEYS
    
    all_contents, all_titles, all_labels = [], [], []
    tr_contents, tr_titles, tr_labels = [], [], []
    va_contents, va_titles, va_labels = [], [], []
    
    random.seed(0)
    for label, k in enumerate(keys):
        all_df[k].fillna('', inplace=True)
        ts = [pre_process(str(t[0]), str(t[1])) for t in all_df[k].values]
        for _ in range(3):
            random.shuffle(ts)
        
        tr_size = round(len(ts) * 0.9)
        # print(f'{len(ts)} vs {len(ts)-tr_size}')
        
        contents, titles = zip(*ts)
        
        all_contents.extend(contents), all_titles.extend(titles), all_labels.extend([label] * len(ts))
        tr_contents.extend(contents[:tr_size]), tr_titles.extend(titles[:tr_size]), tr_labels.extend([label] * tr_size)
        va_contents.extend(contents[tr_size:]), va_titles.extend(titles[tr_size:]), va_labels.extend([label] * (len(ts) - tr_size))
    
    return (
        (all_contents, all_titles, all_labels, va_contents, va_titles, va_labels) if full_train else
        (tr_contents, tr_titles, tr_labels, va_contents, va_titles, va_labels)
    )


def read_test_xlsx():
    test_df = pd.read_excel(test_file, engine='openpyxl')
    test_df.fillna('', inplace=True)
    test_contents, test_titles = zip(*[pre_process(c, t) for c, t in zip(test_df['content'], test_df['title'])])
    return test_contents, test_titles


def save_test_xlsx(labels):
    test_df: pd.DataFrame = pd.read_excel(test_file, engine='openpyxl')
    test_df['channelName'] = labels_to_strs(labels)
    test_df.to_excel(save_file, engine='openpyxl', sheet_name='类别', index=False)


def __get_inp_mask_tar(tokenizer, using_content, contents: List[str], titles: List[str], labels: List[int] = None):
    import torch
    if using_content:
        data_dict = tokenizer(titles, contents, padding=True, truncation=False, return_tensors='pt')
        input_ids, token_type_ids, attention_mask = data_dict['input_ids'], data_dict['token_type_ids'], data_dict['attention_mask']
    else:
        data_dict = tokenizer(titles, padding=True, truncation=False, return_tensors='pt')
        input_ids, token_type_ids, attention_mask = data_dict['input_ids'], None, data_dict['attention_mask']
    
    if labels is None:
        labels = torch.zeros(input_ids.shape[0], dtype=torch.long)
    else:
        labels = torch.tensor(labels, dtype=torch.long)
    labels = labels.reshape(-1).long()
    
    return input_ids, token_type_ids, attention_mask, labels


def get_dataloader(dist, using_content, contents, titles, labels, tokenizer, train, bs):
    from torch.utils.data import TensorDataset, DataLoader
    
    inp, tok, msk, tar = __get_inp_mask_tar(tokenizer, using_content, contents, titles, labels)
    dataset = TensorDataset(inp, tok, msk, tar) if using_content else TensorDataset(inp, msk, tar)
    if train:
        sp = DistributedSampler(dataset, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=bs, sampler=sp, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    else:
        sp, loader = None, DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    return sp, loader


def test_data():
    global train_file, test_file
    train_file = 'raw_train.xlsx'
    test_file = 'test.xlsx'
    tr_contents, tr_titles, tr_labels, va_contents, va_titles, va_labels = read_train_xlsx(is_tuning_hyper_parameters=False)
    print(f'len(tt)={len(tr_titles)}, tt[-1]={tr_titles[-1]}, len(vt)={len(va_titles)}, vt[-1]={va_titles[-1]}')
    test_contents, test_titles = read_test_xlsx()
    # print(type(ls), type(ls[0]), len(ls), ls[0], ls[-1], sep='\n')
    import os
    from transformers import BertTokenizer
    ckpt_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'huggingface', 'hfl', 'chinese-macbert-base'))
    tokenizer = BertTokenizer.from_pretrained(ckpt_path)
    train_dict = tokenizer(tr_titles, tr_contents, padding=True, truncation=False, return_tensors="pt")
    val_dict = tokenizer(va_titles, va_contents, padding=True, truncation=False, return_tensors="pt")
    train_input_ids, val_input_ids = train_dict['input_ids'], val_dict['input_ids']
    train_token_type_ids, val_token_type_ids = train_dict['token_type_ids'], val_dict['token_type_ids']
    train_masks, val_masks = train_dict['attention_mask'], val_dict['attention_mask']
    print(train_input_ids.shape)
    print(train_token_type_ids.shape)
    print(train_masks.shape)
    print(val_input_ids.shape)
    print(val_token_type_ids.shape)
    print(val_masks.shape)


if __name__ == '__main__':
    test_data()
