from transformers import BertTokenizer, BertModel
import os

import torch

from data import NUM_CLASSES


class MacBertCls(torch.nn.Module):
    def __init__(self, extractor, num_classes, dropout_rate=0.4):
        super(MacBertCls, self).__init__()
        self.bert = extractor
        
        if dropout_rate > 1e-3:
            self.dropout = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.svm = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        if self.dropout is not None:
            x = self.dropout(x)
        logits = self.svm(x)
        return logits


def get_tok_model(dropout_rate):
    ckpt_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'huggingface', 'luhua', 'chinese_pretrain_mrc_macbert_large'))
    tokenizer = BertTokenizer.from_pretrained(ckpt_path)
    extractor = BertModel.from_pretrained(ckpt_path)

    return tokenizer, MacBertCls(extractor, NUM_CLASSES, dropout_rate)
