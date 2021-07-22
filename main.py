import argparse
import os
import time
from datetime import datetime
from pprint import pformat

import numpy as np
import colorama
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import AdamW

from adv import FGM
from cfg import parse_cfg
from data import read_train_xlsx, NUM_CLASSES, get_dataloader
from dist import TorchDistManager
from ema import EMA
from log import create_loggers
from loss import LabelSmoothFocalLossV2
from model import get_tok_model
from test import eval_model
from utils import TopKHeap, adjust_learning_rate, AverageMeter, master_echo


def main():
    colorama.init(autoreset=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    parser = argparse.ArgumentParser(description='softbei cls -- news cls task')
    parser.add_argument('--main_py_rel_path', type=str, required=True)
    parser.add_argument('--exp_dirname', type=str, required=True)
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()
    
    sh_root = os.getcwd()
    exp_root = os.path.join(sh_root, args.exp_dirname)
    os.chdir(args.main_py_rel_path)
    prj_root = os.getcwd()
    os.chdir(sh_root)
    
    dist = TorchDistManager(args.exp_dirname, 'auto', 'auto')
    loggers = create_loggers(prj_root, sh_root, exp_root, dist)
    
    cfg = parse_cfg(args.cfg)
    if dist.is_master():
        try:
            main_process(exp_root, cfg, dist, loggers)
        except Exception as e:
            loggers[1].log(pr=-1., rem=0)
            raise e
    else:
        try:
            main_process(exp_root, cfg, dist, loggers)
        except Exception:
            exit(-1)


def main_process(exp_root, cfg, dist, loggers):
    loggers[0].info(f'=> [final cfg]:\n{pformat(dict(cfg))}')
    loggers[1].log(
        tune=cfg.is_tuning_hp,
        dr=cfg.dropout_rate,
        bs=cfg.batch_size, ep=cfg.epochs,
        lr=cfg.lr, wd=cfg.wd, ls=cfg.smooth_ratio, clp=cfg.grad_clip,
        fgm=cfg.fgm, ema=cfg.ema_mom,
        pr=0, rem=0, beg_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )
    
    train_model(exp_root, cfg, dist, loggers)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    decay_parameters = []
    for name, child in model.named_children():
        decay_parameters += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    decay_parameters += list(model._parameters.keys())
    return decay_parameters


def build_op(model, lr, wd):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if 'bias' not in name]
    pg = [{
        "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
        "weight_decay": wd,
    }, {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
        "weight_decay": 0.0,
    }]
    return AdamW(pg, lr=lr, weight_decay=wd)


def train_model(exp_root, cfg, dist, loggers):
    lg, st_lg, tb_lg = loggers
    
    tokenizer, model = get_tok_model(cfg.dropout_rate)
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.dev_idx], output_device=dist.dev_idx)
    
    tr_contents, tr_titles, tr_labels, va_contents, va_titles, va_labels = read_train_xlsx(cfg.is_tuning_hp)
    _, va_ld = get_dataloader(dist, cfg.using_content, va_contents, va_titles, va_labels, tokenizer, train=True, bs=cfg.batch_size * 2)
    cfg.batch_size //= dist.world_size
    tr_sp, tr_ld = get_dataloader(dist, cfg.using_content, tr_contents, tr_titles, tr_labels, tokenizer, train=True, bs=cfg.batch_size)
    
    ema = EMA(model, cfg.ema_mom)
    fgm = FGM(model, cfg.fgm)
    crit = LabelSmoothFocalLossV2(NUM_CLASSES, cfg.smooth_ratio, cfg.alpha, cfg.gamma).cuda()
    
    # todo: frozen
    op = build_op(model, cfg.lr, cfg.wd)
    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    tr_iters = len(tr_ld)
    max_ep = cfg.epochs
    max_iter = max_ep * tr_iters
    saved_af1, best_af1, best_af1_ema = -1, -1, -1
    topk_af1s, topk_af1s_ema = TopKHeap(maxsize=6), TopKHeap(maxsize=6)
    epoch_speed = AverageMeter(3)
    
    te_freq = max(1, round(tr_iters // 3))
    saved_path = os.path.join(exp_root, f'best_ckpt.pth')
    loop_start_t = time.time()
    for ep in range(max_ep):
        tr_sp.set_epoch(ep)
        ep_str = f'%{len(str(max_ep))}d'
        ep_str %= ep + 1
        ep_str = f'ep[{ep_str}/{max_ep}]'
        
        torch.cuda.empty_cache()
        master_echo(dist.is_master(), f' @@@@@ {exp_root}      be={best_af1:5.2f} ({best_af1_ema:5.2f})', '36')
        
        ep_start_t = time.time()
        last_t = time.time()
        for it, data in enumerate(tr_ld):
            it_str = f'%{len(str(tr_iters))}d'
            it_str %= it + 1
            it_str = f'it[{it_str}/{tr_iters}]'
            cur_iter = it + ep * tr_iters
            data_t = time.time()
            
            if cfg.using_content:
                inp, tok, msk, tar = data
                inp, tok, msk, tar = inp.cuda(non_blocking=True), tok.cuda(non_blocking=True), msk.cuda(non_blocking=True), tar.cuda(non_blocking=True)
            else:
                inp, msk, tar = data
                inp, tok, msk, tar = inp.cuda(non_blocking=True), None, msk.cuda(non_blocking=True), tar.cuda(non_blocking=True)
            if cur_iter == 0:
                lg.info(f'inp.shape={inp.shape}')
            cuda_t = time.time()
            
            logits = model(inp, tok, msk)
            loss = crit(logits, tar)
            forw_t = time.time()
            
            loss.backward()
            back_t = time.time()
            
            orig_norm = float(torch.nn.utils.clip_grad_norm_(all_params, float(cfg.grad_clip)))
            clip_t = time.time()

            if fgm.open():
                fgm.attack()
                op.zero_grad() # 如果不想累加梯度，就把这里的注释取消
                logits = model(inp, tok, msk)
                loss = crit(logits, tar)
                loss.backward()
                fgm.restore()
            
            if cur_iter % 30 == 0:
                preds = logits.detach().argmax(dim=1)
                tr_acc = 100. * preds.eq(tar).sum().item() / tar.shape[0]
                tr_loss = loss.item()
                
                tb_lg.add_scalar('tr/acc', tr_acc, cur_iter)
                tb_lg.add_scalar('tr/loss', tr_loss, cur_iter)
            
            sche_lr = adjust_learning_rate(op, cur_iter, max_iter, cfg.lr)
            actual_lr = sche_lr * min(1., float(cfg.grad_clip) / orig_norm)
            op.step()
            op.zero_grad()
            optm_t = time.time()
            
            ema.step(model, cur_iter + 1)
            
            logging = cur_iter == max_iter - 1 or cur_iter % te_freq == 0
            if logging or orig_norm > 8 or cur_iter < tr_iters:
                tb_lg.add_scalars('opt/lr', {'sche': sche_lr, 'actu': actual_lr}, cur_iter)
                tb_lg.add_scalars('opt/norm', {'orig': orig_norm, 'clip': cfg.grad_clip}, cur_iter)
            
            if logging:
                preds = logits.detach().argmax(dim=1)
                tr_acc = 100. * preds.eq(tar).sum().item() / tar.shape[0]
                tr_loss = loss.item()
                
                va_acc, va_rec, va_af1, va_loss = eval_model(cfg.using_content, va_ld, model)
                topk_af1s.push_q(va_af1)
                best_af1 = max(best_af1, va_af1)
                
                ema.load_ema(model)
                va_acc_ema, va_rec_ema, va_af1_ema, va_loss_ema = eval_model(cfg.using_content, va_ld, model)
                best_af1_ema = max(best_af1_ema, va_af1_ema)
                if best_af1_ema > saved_af1:
                    saved_af1 = best_af1_ema
                    torch.save(model.module.state_dict(), saved_path)
                ema.recover(model)
                topk_af1s_ema.push_q(va_af1_ema)
                
                va_t = time.time()
                
                if best_af1 > saved_af1:
                    saved_af1 = best_af1
                    torch.save(model.module.state_dict(), saved_path)
                
                remain_time, finish_time = epoch_speed.time_preds(max_ep - (ep + 1))
                
                lg.info(
                    f'=> {ep_str}, {it_str}:    lr={sche_lr:.3g}({actual_lr:.3g}), nm={orig_norm:.1f}\n'
                    f'  [tr] L={tr_loss:.3f}, acc={tr_acc:5.2f}, da={data_t - last_t:.3f} cu={cuda_t - data_t:.3f} fp={forw_t - cuda_t:.3f} bp={back_t - forw_t:.3f} cl={clip_t - back_t:.3f} op={optm_t - clip_t:.3f} te={va_t - optm_t:.3f}\n'
                    f'  [va] L={va_loss:.3f}({va_loss_ema:.3f}), f1={va_af1:5.2f}({va_af1_ema:5.2f}), ac={va_acc:5.2f}({va_acc_ema:5.2f}), re={va_rec:5.2f}({va_rec_ema:5.2f})      remain [{str(remain_time)}] ({finish_time})       >>> [best]={best_af1:5.2f}({best_af1_ema:5.2f})'
                )
                tb_lg.add_scalar('va/macro_F1', va_af1, cur_iter)
                tb_lg.add_scalars('va/macro_F1', {'ema': va_af1_ema}, cur_iter)
                tb_lg.add_scalar('va/acc', va_acc, cur_iter)
                tb_lg.add_scalars('va/acc', {'ema': va_acc_ema}, cur_iter)
                tb_lg.add_scalar('va/rec', va_rec, cur_iter)
                tb_lg.add_scalars('va/rec', {'ema': va_rec_ema}, cur_iter)
                tb_lg.add_scalar('va/loss', va_loss, cur_iter)
                tb_lg.add_scalars('va/loss', {'ema': va_loss_ema}, cur_iter)
                
                st_lg.log(
                    pr=(cur_iter + 1) / max_iter,
                    clr=sche_lr, nm=orig_norm,
                    tr_L=tr_loss, te_L=va_loss, em_L=va_loss_ema,
                    tr_A=tr_acc, te_F=va_af1, em_F=va_af1_ema, te_A=va_acc, em_A=va_acc_ema, te_R=va_rec, em_R=va_rec_ema,
                    be=best_af1, be_e=best_af1_ema,
                    rem=remain_time.seconds, end_t=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_time.seconds)),
                )
            
            last_t = time.time()
        # iteration loop end
        epoch_speed.update(time.time() - ep_start_t)
        tb_lg.add_scalar('va_ep_best/macro_F1', best_af1, ep + 1)
        tb_lg.add_scalars('va_ep_best/macro_F1', {'ema': best_af1_ema}, ep + 1)
    
    # epoch loop end
    
    topk_va_af1 = sum(topk_af1s) / len(topk_af1s)
    topk_va_af1_ema = sum(topk_af1s_ema) / len(topk_af1s_ema)
    
    topk_af1s = dist.dist_fmt_vals(topk_va_af1, None)
    topk_af1s_ema = dist.dist_fmt_vals(topk_va_af1_ema, None)
    best_af1s = dist.dist_fmt_vals(best_af1, None)
    best_af1s_ema = dist.dist_fmt_vals(best_af1_ema, None)
    if dist.is_master():
        [tb_lg.add_scalar('z_final_best/topk_macro_F1', topk_af1s.max().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_best/topk_macro_F1_ema', topk_af1s_ema.max().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_best/best_macro_F1', best_af1s.max().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_best/best_macro_F1_ema', best_af1s_ema.max().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_mean/topk_macro_F1', topk_af1s.mean().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_mean/topk_macro_F1_ema', topk_af1s_ema.mean().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_mean/best_macro_F1', best_af1s.mean().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_mean/best_macro_F1_ema', best_af1s_ema.mean().item(), e) for e in [-max_ep, max_ep]]
    
    eval_str = (
        f' mean-top     @ (max={topk_af1s.max():5.2f}, mean={topk_af1s.mean():5.2f}, std={topk_af1s.std():.2g}) {str(topk_af1s).replace(chr(10), " ")})\n'
        f' EMA mean-top @ (max={topk_af1s_ema.max():5.2f}, mean={topk_af1s_ema.mean():5.2f}, std={topk_af1s_ema.std():.2g}) {str(topk_af1s_ema).replace(chr(10), " ")})\n'
        f' best         @ (max={best_af1s.max():5.2f}, mean={best_af1s.mean():5.2f}, std={best_af1s.std():.2g}) {str(best_af1s).replace(chr(10), " ")})\n'
        f' EMA best     @ (max={best_af1s_ema.max():5.2f}, mean={best_af1s_ema.mean():5.2f}, std={best_af1s_ema.std():.2g}) {str(best_af1s_ema).replace(chr(10), " ")})'
    )
    
    dt = time.time() - loop_start_t
    lg.info(
        f'=> training finished,'
        f' total time cost: {dt / 60:.2f}min ({dt / 60 / 60:.2f}h)\n'
        f' performance: \n{eval_str}'
    )
    
    st_lg.log(
        pr=1., rem=0,
        m_tk=topk_af1s.mean().item(), m_tk_e=topk_af1s_ema.mean().item(),
        m_be=best_af1s.mean().item(), m_be_e=best_af1s_ema.mean().item(),
        end_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )
    
    dist.barrier()
    tb_lg.close()


if __name__ == '__main__':
    import numpy as np
    from torch import multiprocessing as mp
    main()
