import torch
import torch.nn.functional as F


class LabelSmoothFocalLoss(torch.nn.Module):
    def __init__(self, num_classes, smooth_ratio=0.05, alpha=None, gamma=2):
        # standard CE: smooth_ratio=0, alpha=None, gamma=0
        super(LabelSmoothFocalLoss, self).__init__()
        
        if smooth_ratio > 0 and num_classes < 10:
            smooth_ratio = 0.005 * num_classes
        
        if alpha is not None:
            alpha = alpha.view(-1).unsqueeze(0).contiguous()
            assert alpha.numel() == num_classes
            alpha /= alpha.sum()
        else:
            alpha = 1
        
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes
        self.alpha, self.gamma = alpha, gamma
    
    def forward(self, logits: torch.Tensor, targets):
        one_hot = torch.ones(logits.shape, device=logits.device) * self.v
        one_hot.scatter_(1, targets.view(-1, 1), 1 - self.smooth_ratio + self.v)
        one_hot *= self.alpha
        one_hot /= one_hot.sum()
        
        prob = F.softmax(logits, dim=1)
        weight = (1 - prob) ** self.gamma
        ce_loss = -(one_hot * prob.log())
        
        focal_loss = (weight * ce_loss).sum()
        return focal_loss


class LabelSmoothFocalLossV2(LabelSmoothFocalLoss):
    # todo: 这个是没有alpha的
    def forward(self, logits, targets):
        one_hot = torch.ones(logits.shape, device=logits.device) * self.v
        one_hot.scatter_(1, targets.view(-1, 1), 1 - self.smooth_ratio + self.v)
        
        log_prob = F.log_softmax(logits, dim=1)
        ce_loss = -(one_hot * log_prob).sum(dim=1)
        weight = (1 - torch.exp(-ce_loss)) ** self.gamma
        
        focal_loss = (ce_loss * weight).mean()
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
    logits = torch.rand(5, 10)
    targets = torch.randperm(10)[:5]
    logits[:, 7] += 0
    targets[...] = 7
    print(targets)
    
    print('CE           =', F.cross_entropy(logits, targets), LabelSmoothFocalLoss(10, 0, None, 0)(logits, targets), LabelSmoothFocalLossV2(10, 0, None, 0)(logits, targets))
    print('CE  + ls0.1  =', _LabelSmoothCELoss(num_classes=10, smooth_ratio=0.1)(logits, targets))
    print('')
    print('FCE          =', __focal_loss(logits, targets))
    print('FCE + ls0    =', LabelSmoothFocalLoss(num_classes=10, smooth_ratio=0.)(logits, targets))
    print('FCE2 + ls0   =', LabelSmoothFocalLossV2(num_classes=10, smooth_ratio=0.)(logits, targets))
    print('FCE + ls0.1  =', LabelSmoothFocalLoss(num_classes=10, smooth_ratio=0.1)(logits, targets))
    print('FCE2 + ls0.1 =', LabelSmoothFocalLossV2(num_classes=10, smooth_ratio=0.1)(logits, targets))
