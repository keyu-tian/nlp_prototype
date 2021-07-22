import os
import sys
import time
import datetime
from pprint import pprint as pp
from pprint import pformat as pf
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class MacBertCls(torch.nn.Module):
    def __init__(self, extractor, num_classes, dropout_rate=0.4):
        super(MacBertCls, self).__init__()
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
    def __init__(self, ckpt_path):
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.tokenizer, self.model = ..., ...
        self.cuda = torch.cuda.is_available()
    
    def initialize(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.ckpt_path)
        self.model = MacBertCls(BertModel.from_pretrained(self.ckpt_path), 16, 0)
        if self.cuda:
            self.model = self.model.cuda()
    
    def infer_one_item(self, title, content):
    
    