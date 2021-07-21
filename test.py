import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from pprint import pprint as pp

from cfg import parse_cfg
from data import read_test_xlsx, get_dataloader, save_test_xlsx
from model import get_tok_model


@torch.no_grad()
def eval_model(using_content, va_ld, model: torch.nn.Module, is_test=False):
    tr = model.training
    model.train(False)
    tot_tar, tot_pred, tot_loss, tot_item = [], [], 0., 0
    for data in va_ld:
    
        if using_content:
            inp, tok, msk, tar = data
            inp, tok, msk, tar = inp.cuda(non_blocking=True), tok.cuda(non_blocking=True), msk.cuda(non_blocking=True), tar.cuda(non_blocking=True)
        else:
            inp, msk, tar = data
            inp, tok, msk, tar = inp.cuda(non_blocking=True), None, msk.cuda(non_blocking=True), tar.cuda(non_blocking=True)
        
        bs = inp.shape[0]
        logits = model(inp, tok, msk)
        tot_tar.append(tar), tot_pred.append(logits.argmax(dim=1))
        tot_loss += F.cross_entropy(logits, tar).item() * bs
        tot_item += bs
    model.train(tr)
    
    tot_tar = torch.cat(tot_tar).cpu().tolist()
    tot_pred = torch.cat(tot_pred).cpu().tolist()
    res = classification_report(tot_tar, tot_pred, output_dict=True)['macro avg']
    
    if is_test:
        pp(res)
        return tot_pred
    else:
        return 100 * res['precision'], 100 * res['recall'], 100 * res['f1-score'], tot_loss / tot_item


def main():
    cfg = parse_cfg('cfg.yaml')

    test_contents, test_titles = read_test_xlsx()
    te_labels = None

    tokenizer, model = get_tok_model(0)
    model.cuda()
    model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
    
    _, te_ld = get_dataloader(None, cfg.using_content, test_contents, test_titles, te_labels, tokenizer, False, 128)
    tot_pred = eval_model(False, te_ld, model, is_test=True)
    save_test_xlsx(tot_pred)
    

if __name__ == '__main__':
    main()
