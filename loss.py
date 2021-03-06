import torch
import torch.nn.functional as F


class LabelSmoothFocalLoss(torch.nn.Module):
    """
    >>> B, NC = 4, 3
    >>> fl = LabelSmoothFocalLoss(num_classes=NC, alpha=[1, 5, 1], gamma=2)
    >>> logits = torch.rand(B, NC)
    >>> targets = torch.randint(NC, size=(B,)).long()
    >>> if torch.cuda.is_available():
    >>>     fl, logits, targets = fl.cuda(), logits.cuda(), targets.cuda()
    >>>
    >>> logits.requires_grad_(True)
    >>> loss = fl(logits, targets)
    >>> loss.backward()
    """
    def __init__(self, num_classes, alpha=None, gamma=2, smoothing=True):
        # standard CE: smooth_ratio=0, alpha=None, gamma=0
        super(LabelSmoothFocalLoss, self).__init__()

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha)
            alpha = alpha.view(-1).unsqueeze(0).contiguous().float()
            assert alpha.numel() == num_classes
            alpha /= alpha.sum()
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = 1
        
        self.smooth_ratio: float = min(0.08, num_classes * 0.004) if smoothing else 0
        self.v: float = self.smooth_ratio / num_classes
        self.gamma: float = gamma
        self.register_buffer('ones', torch.ones(1, num_classes) * self.v)
    
    def forward(self, logits, targets):
        prob = F.softmax(logits, dim=1)     # (B, C)
        focal = (1 - prob) ** self.gamma    # (B, C)
        
        one_hot = self.ones.expand(logits.shape)
        one_hot = self.alpha * one_hot.scatter(1, targets.view(-1, 1), 1 - self.smooth_ratio + self.v)
        one_hot /= one_hot.sum()            # (B, C)
        
        ce_loss = -(one_hot * prob.log())   # (B, C)
        
        focal_loss = (focal * ce_loss).sum()    # (B, C).sum()
        return focal_loss


class LabelSmoothFocalLossV2(LabelSmoothFocalLoss):
    # todo: 这个是没有alpha的
    def forward(self, logits, targets):
        one_hot = torch.ones(logits.shape, device=logits.device) * self.v
        one_hot.scatter_(1, targets.view(-1, 1), 1 - self.smooth_ratio + self.v)    # (B, C)
        
        log_prob = F.log_softmax(logits, dim=1)     # (B, C)
        ce_loss = -(one_hot * log_prob).sum(dim=1)  # (B,)
        focal = (1 - torch.exp(-ce_loss)) ** self.gamma    # (B,)
        
        focal_loss = (focal * ce_loss).mean()      # (B,).mean()
        return focal_loss


class _LabelSmoothCELoss(torch.nn.Module):
    def __init__(self, smooth_ratio, num_classes):
        super(_LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes
        
        self.logsoft = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, logits, targets):
        one_hot = torch.zeros_like(logits)
        one_hot.fill_(self.v)
        y = targets.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1 - self.smooth_ratio + self.v)
        
        loss = - torch.sum(self.logsoft(logits) * (one_hot.detach())) / logits.size(0)
        return loss


def __focal_loss(logits, targets, alpha=1, gamma=2):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  # important to add reduction='none' to keep per-batch-item loss
    focal_loss = (alpha * (1 - torch.exp(-ce_loss)) ** gamma * ce_loss).mean()
    return focal_loss


if __name__ == '__main__':
    torch.set_printoptions(precision=10)
    
    logits = torch.rand(5, 10)
    targets = torch.randperm(10)[:5]
    logits[:, 7] += -5
    targets[...] = 7
    print(targets)
    
    print('CE           =', F.cross_entropy(logits, targets), LabelSmoothFocalLoss(10, None, 0)(logits, targets), LabelSmoothFocalLossV2(10, None, 0)(logits, targets))
    print('CE  + ls0.1  =', _LabelSmoothCELoss(num_classes=10, smooth_ratio=0.1)(logits, targets))
    print('')
    print('FCE          =', __focal_loss(logits, targets))
    print('FCE + ls0    =', LabelSmoothFocalLoss(num_classes=10, alpha=[1]*10)(logits, targets))
    print('FCE2 + ls0   =', LabelSmoothFocalLossV2(num_classes=10)(logits, targets))
    print('')
    print('FCE + ls0.1  =', LabelSmoothFocalLoss(num_classes=10)(logits, targets))
    print('FCE2 + ls0.1 =', LabelSmoothFocalLossV2(num_classes=10)(logits, targets))
