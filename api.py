import os
import sys
import time
import datetime
from pprint import pprint as pp
from pprint import pformat as pf
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class _MacBertCls(torch.nn.Module):
    def __init__(self, extractor, num_classes, dropout_rate=0.4):
        super(_MacBertCls, self).__init__()
        self.bert: BertModel = extractor
        
        if dropout_rate > 1e-3:
            self.dropout = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.svm = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        if token_type_ids is None:
            x = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        else:
            x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)['pooler_output']
        
        if self.dropout is not None:
            x = self.dropout(x)
        logits = self.svm(x)
        return logits


class NewsClassifier(object):
    CLS_KEYS = ['体育', '其他健康', '其他宠物', '其他慈善', '其他政治', '其他文艺', '其他旅游', '其他育儿', '军事', '娱乐', '房产', '教育', '汽车', '游戏', '科技', '财经']
    NUM_CLASSES = len(CLS_KEYS)

    def __init__(self, ckpt_path):
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.tokenizer, self.model = ..., ...
        self.cuda = torch.cuda.is_available()
    
    def initialize(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.ckpt_path)
        self.model = _MacBertCls(BertModel.from_pretrained(self.ckpt_path), NewsClassifier.NUM_CLASSES, 0)
        ckpt = torch.load(os.path.join(self.ckpt_path, 'ckpt.pth'), map_location='cpu')
        self.model.load_state_dict()
        if self.cuda:
            self.model = self.model.cuda()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
    
    def infer_for_one_item(self, title: str, content: str) -> str:
        data_dict = self.tokenizer([title], [content], padding=True, truncation=False, return_tensors='pt')
        input_ids, token_type_ids, attention_mask = data_dict['input_ids'], data_dict['token_type_ids'], data_dict['attention_mask']
        self.model(input_ids, token_type_ids, attention_mask)
    
    