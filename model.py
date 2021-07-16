from transformers import BertTokenizer, BertModel
import os

import torch
import torch.nn as nn

from data import NUM_CLASSES


class MacBertCls(nn.Module):
    def __init__(self, extractor, num_classes, dropout_rate=0.4):
        global model
        super(MacBertCls, self).__init__()
        self.bert = extractor
        
        self.dropout = nn.Dropout(dropout_rate)
        self.svm = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        dropout_output = self.dropout(pooled_output)
        logits = self.svm(dropout_output)
        return logits


def get_tok_model(dropout_rate):
    ckpt_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'huggingface', 'luhua', 'chinese_pretrain_mrc_macbert_large'))
    tokenizer = BertTokenizer.from_pretrained(ckpt_path)
    extractor = BertModel.from_pretrained(ckpt_path)

    return tokenizer, MacBertCls(extractor, NUM_CLASSES, dropout_rate)