import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time




def init_class_prototypes(args, model, loader):
    """Initialize class prototypes"""
    model.eval()
    start = time.time()
    prototype_counts = [0]*args.n_cls
    with torch.no_grad():
        prototypes = torch.zeros(args.n_cls, args.feat_dim).cuda()
        for i, (input, target) in enumerate(loader):
            input, target = input.cuda(), target.cuda()
            features = model(input)
            for j, feature in enumerate(features):
                prototypes[target[j].item()] += feature
                prototype_counts[target[j].item()] += 1
        for cls in range(args.n_cls):
            prototypes[cls] /=  prototype_counts[cls] 
        # measure elapsed time
        duration = time.time() - start
        print(f'Time to initialize prototypes: {duration:.3f}')
        prototypes = F.normalize(prototypes, dim=1)
        return prototypes.detach()
    

class ASLoss(nn.Module):

    def __init__(self, args, temperature= 0.1, base_temperature=0.1):
        super(ASLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, prototypes):
        norms = torch.norm(prototypes,dim=1,keepdim=True).detach()
        norm_prototypes = prototypes / norms    
        num_cls = self.args.n_cls
        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.T).float()).cuda()
        logits = torch.div(
            (torch.matmul(norm_prototypes, norm_prototypes.T)- 1/(1-num_cls) ) ** 2,
            self.temperature).cuda()

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

def get_prototypes(args, model, loader, op_epoch):
    # torch.cuda.set_device(args.gpu)
    prototypes=init_class_prototypes(args, model, loader)
    prototypes.requires_grad_(True).cuda()
    optimizer=optim.Adam([prototypes], lr=0.01)
    # losses=prototypeLoss().cuda()
    losses = ASLoss(args).cuda() 
    start = time.time()
    for i in range(op_epoch):
        optimizer.zero_grad()
        loss=losses(prototypes)
        loss.backward()
        optimizer.step()
        if loss==0:
            break
    prototypes=F.normalize(prototypes,dim=1)
    duration = time.time() - start
    print(f'Time to update prototypes: {duration:.3f}')
    return prototypes.data
if __name__ =="__name__":
    a=0
    # prototypes = get_prototypes()
