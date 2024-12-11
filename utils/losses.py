
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class PECLoss(nn.Module):
    def __init__(self, args, prototypes, temperature=0.07, base_temperature=0.05):
        super(PECLoss, self).__init__()
        self.args = args
        self.prototypes = prototypes 
        self.temperature = temperature
        self.base_temperature = base_temperature
        

    def forward(self, features, labels):
        proxy_labels = torch.arange(0, self.args.n_cls).cuda()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, proxy_labels.T).float().cuda() #bz, cls
        feat_dot_prototype = torch.div(
            torch.matmul(features, self.prototypes.T),
            self.temperature)

        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss
    



class ASLoss(nn.Module):

    def __init__(self, args, temperature= 0.1, base_temperature=0.1):
        super(ASLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, prototypes):
        norms = torch.norm(prototypes,dim=1,keepdim=True).detach()
        norm_prototypes=prototypes / norms    
        num_cls = self.args.n_cls
        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.T).float()).cuda()
        logits = torch.div(
            torch.matmul(norm_prototypes, norm_prototypes.T),
            self.temperature).cuda()
        logits = (logits - 1/(1-num_cls)) ** 2

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_cls).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss



