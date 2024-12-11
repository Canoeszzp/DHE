import argparse
import math
import os
import time
from datetime import datetime
import logging
import tensorboard_logger as tb_logger
import pprint

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import numpy as np

from utils import (PECLoss, AverageMeter, adjust_learning_rate, warmup_learning_rate, 
                set_loader_small, set_loader_ImageNet, set_model)
from utils.prototypes import get_prototypes

parser = argparse.ArgumentParser(description='Training with DHE')
parser.add_argument('--gpu', default=0, type=int, help='which GPU to use')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--feat_dim', default = 128, type=int,
                    help='feature dim')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--id_loc', default="./data", type=str, help='location of in-distribution dataset')
parser.add_argument('--model', default='resnet34', type=str, help='model architecture: [resnet18, resnet34, resnet50]')
parser.add_argument('--head', default='mlp', type=str, help='mlp linear')
parser.add_argument('--loss', default = 'DHE', type=str, help='train loss')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
parser.add_argument('--save-epoch', default=100, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default= 512, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', default=0.5, type=float,
                    help='initial learning rate')
# if linear lr schedule
parser.add_argument('--lr_decay_epochs', type=str, default='100,150,180,300',
                        help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
# if cosine lr schedule
parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
parser.add_argument('--normalize', action='store_true',
                        help='normalize feat embeddings')
parser.add_argument('--subset', default=False,
                        help='whether to use subset of training set to init prototypes')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%d_%m_%H:%M")

#processing str to list for linear lr scheduling
args.lr_decay_epochs = [int(step) for step in args.lr_decay_epochs.split(',')]


if args.in_dataset == "CIFAR-10":
    args.n_cls = 10
elif args.in_dataset in ["CIFAR-100", "ImageNet-100"]:
    args.n_cls = 100

args.name = (f"{date_time}_DHE_{args.model}_lr_{args.learning_rate}_cosine_"
        f"{args.cosine}_bsz_{args.batch_size}_{args.loss}_wd_{args.w}_{args.epochs}_{args.feat_dim}_"
        f"trial_{args.trial}_temp_{args.temp}_{args.in_dataset}_pm_{args.proto_m}")

args.log_directory = "logs/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
args.model_directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name= args.name )
args.tb_path = './save/DHE/{}_tensorboard'.format(args.in_dataset)
prototypes_root='./prototypes/{}/'.format(args.in_dataset)
if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)
args.tb_folder = os.path.join(args.tb_path, args.name)
if not os.path.isdir(args.tb_folder):
    os.makedirs(args.tb_folder)

    

#save args
with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

#init log
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s : %(message)s')
fileHandler = logging.FileHandler(os.path.join(args.log_directory, "train_info.log"), mode='w')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.addHandler(fileHandler)
log.addHandler(streamHandler) 

log.debug(state)

#set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
log.debug(f"{args.name}")

# warm-up for large-batch training
if args.batch_size > 256:
    args.warm = True
if args.warm:
    args.warmup_from = 0.001
    args.warm_epochs = 10
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
    else:
        args.warmup_to = args.learning_rate


def main():
    tb_log = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    if args.in_dataset == "ImageNet-100":
        train_loader, val_loader = set_loader_ImageNet(args)
        aux_loader, _  = set_loader_ImageNet(args, eval = True)
    else:
        train_loader, val_loader = set_loader_small(args)
        aux_loader, _ = set_loader_small(args, eval = True)

    model = set_model(args)

    prototypes = get_prototypes(args, model, aux_loader, 500).cuda()
    criterion_PEC=PECLoss(args, prototypes, temperature=args.temp).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        # train for one epoch
        PEC_loss = train_DHE(args, train_loader, model, criterion_PEC, optimizer, epoch, log)

        tb_log.log_value('train_PEC_loss',PEC_loss,epoch)

        # tensorboard logger
        tb_log.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # save checkpoint
        if (epoch + 1) % args.save_epoch == 0: 
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'ours_state_dict': criterion_PEC.state_dict(),}, epoch + 1)

def train_DHE(args, train_loader, model, criterion_PEC, optimizer, epoch, log):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    PEC_losses = AverageMeter()


    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        warmup_learning_rate(args, epoch, i, len(train_loader), optimizer)
        bsz = target.shape[0]
        input = torch.cat([input[0], input[1]], dim=0).cuda()
        target = target.repeat(2).cuda()

        penultimate = model.encoder(input).squeeze()
        if args.normalize: 
            penultimate= F.normalize(penultimate, dim=1)
        features= model.head(penultimate)
        features= F.normalize(features, dim=1)
        PEC_loss = criterion_PEC(features,target)
        PEC_losses.update(PEC_loss.data, input.size(0))
        loss=PEC_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            log.debug('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'PEC Loss {PECloss.val:.4f} ({PEC.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time, PECloss=PEC_losses)) 
                
    return PEC_losses.avg


def save_checkpoint(args, state, epoch):
    """Saves checkpoint to disk"""
    filename = args.model_directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)


if __name__ == '__main__':
    main()
