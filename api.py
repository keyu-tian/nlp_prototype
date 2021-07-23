import os
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


class _MacBertCls(torch.nn.Module):
    def __init__(self, extractor, num_classes):
        super(_MacBertCls, self).__init__()
        self.bert: BertModel = extractor
        
        self.svm = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        if token_type_ids is None:
            x = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        else:
            x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)['pooler_output']
        logits = self.svm(x)
        return logits


class NewsClassifier(object):
    CLS_KEYS = ['体育', '其他健康', '其他宠物', '其他慈善', '其他政治', '其他文艺', '其他旅游', '其他育儿', '军事', '娱乐', '房产', '教育', '汽车', '游戏', '科技', '财经']
    NUM_CLASSES = len(CLS_KEYS)

    def __init__(self, ckpt_path):
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.tokenizer, self.model = ..., ...
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            _ = torch.empty(10).cuda()
    
    def initialize(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.ckpt_path)
        self.model = _MacBertCls(BertModel.from_pretrained(self.ckpt_path), NewsClassifier.NUM_CLASSES)
        self.model.svm.load_state_dict(torch.load(os.path.join(self.ckpt_path, 'linear.pth'), map_location='cpu'))
        if self.cuda:
            self.model = self.model.cuda()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def infer_one_item(self, title: str, content: str) -> str:
        title, content = NewsClassifier._pre_process(title=title, content=content)
        data_dict = self.tokenizer([title], [content], padding=True, truncation=False, return_tensors='pt')
        input_ids, token_type_ids, attention_mask = data_dict['input_ids'], data_dict['token_type_ids'], data_dict['attention_mask']
        if self.cuda:
            input_ids, token_type_ids, attention_mask = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda()
        logits = self.model(input_ids, token_type_ids, attention_mask)
        pred = NewsClassifier.CLS_KEYS[logits[0].cpu().argmax().item()]
        if pred.startswith('其他'):
            pred = '其他'
        return pred

    @torch.no_grad()
    def infer_items(self, titles: List[str], contents: List[str]) -> List[str]:
        titles, contents = zip(*[NewsClassifier._pre_process(t, c) for t, c in zip(titles, contents)])
        data_dict = self.tokenizer(titles, contents, padding=True, truncation=False, return_tensors='pt')
        batch_size = 128 if self.cuda else 4
        input_ids, token_type_ids, attention_mask = data_dict['input_ids'], data_dict['token_type_ids'], data_dict['attention_mask']
        loader = DataLoader(
            TensorDataset(input_ids, token_type_ids, attention_mask),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        tot_pred = []
        bar = tqdm(enumerate(loader), unit=f'每{batch_size}条', dynamic_ncols=True)
        for i, (inp, tok, msk) in bar:
            if self.cuda:
                inp, tok, msk = inp.cuda(non_blocking=True), tok.cuda(non_blocking=True), msk.cuda(non_blocking=True)
            logits = self.model(inp, tok, msk)
            preds = logits.argmax(dim=1)
            tot_pred.append(preds)
            bar.set_postfix_str(f'第{i*batch_size}条/共{input_ids.shape[0]}条，当前类别={NewsClassifier.CLS_KEYS[preds[0].item()]}')
        tot_pred = torch.cat(tot_pred).cpu().tolist()
        return NewsClassifier._labels_to_strs(tot_pred)

    @staticmethod
    def _pre_process(title: str, content: str):
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
        return t, c
    
    @staticmethod
    def _labels_to_strs(indices) -> List[str]:
        labels = []
        for i in indices:
            if NewsClassifier.CLS_KEYS[i].startswith('其他'):
                labels.append('其他')
            else:
                labels.append(NewsClassifier.CLS_KEYS[i])
        return labels


if __name__ == '__main__':
    cls = NewsClassifier('./chinese-macbert-base')
    cls.initialize()
    print(cls.infer_one_item('纽约油价飙升', '今日纽约油价飙升啊喂'))
    print(cls.infer_items(
        titles=['我是财经新闻标题', '我是体育新闻'*60],
        contents=['我是财经新闻正文', '我是体育新闻正文'*60],
    ))
