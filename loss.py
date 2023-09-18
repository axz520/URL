import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class CELoss(nn.Module):
    def __init__(self, num_classes=5):
        super(CELoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, labels, weight=None, t=1.):
        pred = F.softmax(pred / t, dim=1)
        pred = torch.clamp(pred, min=1e-6, max=1.0)
        if(weight is None):
            loss = -torch.sum(torch.log(pred) * labels) / pred.shape[0]
        else:
            loss = -torch.sum(torch.log(pred) * labels * weight.unsqueeze(1)) / pred.shape[0]
        return loss
    

class OrdinaryReg(nn.Module):
    def __init__(self, limit=0.01) -> None:
        super(OrdinaryReg, self).__init__()
        self.limit = limit

    def forward(self, matrix, y=None):
        margin = torch.tensor(self.limit).to(matrix.device)
        if(y == None):
            y = matrix.argmax(dim=1)
        K = matrix.shape[1]
        loss = 0.
        for k in range(K-1):
            reg_gt = (y >= k+1).float() * F.relu(margin + matrix[:,  k] - matrix[:, k+1])
            reg_lt = (y <= k).float() * F.relu(margin + matrix[:, k+1] - matrix[:, k])
            loss += torch.mean(reg_gt + reg_lt)
        return loss
    

# followed by Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, 
# D.: Supervised contrastive learning. Advances in neural information processing systems 33, 18661-18673 (2020)
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0)
        mask = mask * logits_mask

        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
