import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from pprint import pprint as pp


from data import read_test_xlsx, get_dataloader, save_test_xlsx
from model import get_tok_model


@torch.no_grad()
def eval_model(va_ld, model: torch.nn.Module, is_test=False):
    tr = model.training
    model.train(False)
    tot_tar, tot_pred, tot_loss, tot_item = [], [], 0., 0
    for (inp, msk, tar) in va_ld:
        bs = inp.shape[0]
        inp, msk, tar = inp.cuda(non_blocking=True), msk.cuda(non_blocking=True), tar.cuda(non_blocking=True)
        logits = model(inp, msk)
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
    te_texts = read_test_xlsx()
    te_labels = None

    tokenizer, model = get_tok_model(0)
    model.cuda()
    model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
    
    _, te_ld = get_dataloader(None, te_texts, te_labels, tokenizer, False, 128)
    tot_pred = eval_model(te_ld, model, is_test=True)
    save_test_xlsx(tot_pred)
    

if __name__ == '__main__':
    main()
