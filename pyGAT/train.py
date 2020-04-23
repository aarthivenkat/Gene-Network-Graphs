from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics

from utils import load_joshi_data#, accuracy
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
# sparse not updated yet
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=1, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_joshi_data("Cd8a")

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=1, # regression
                # nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(ncells = features.shape[0], #features is cells x gene x gene features
                ngenes=features.shape[1], 
                nfeat=features.shape[2],
                nhid=args.hidden, 
                nclass=1, # regression
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr) 
                       # weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

def loss_fn(output, labels, attention, lam):
    reconstruction = F.smooth_l1_loss(output, labels.float()).float()
    sparsity = lam * (torch.norm(attention, 0) / labels.shape[0])
    reg = 0.
    for param in model.parameters():
        reg += torch.norm(param, 0)

    return (reconstruction + sparsity)
    # return (reconstruction + reg)  

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, attention = model(features, adj)
    
    output = output.reshape(-1)

    loss_train = loss_fn(output[idx_train], labels[idx_train], attention[idx_train], 0.05)
    r2_train = sklearn.metrics.r2_score(labels[idx_train].cpu(), output[idx_train].cpu().detach().numpy())
    l0_norm_att_train = torch.norm(attention[idx_train], 0) / len(idx_train)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output, attention = model(features, adj)
        output = output.reshape(-1)

    loss_val = loss_fn(output[idx_val], labels[idx_val], attention[idx_val], 0.05)
    r2_val = sklearn.metrics.r2_score(labels[idx_val].cpu(), output[idx_val].cpu().detach().numpy())
    l0_norm_att_val = torch.norm(attention[idx_val], 0) / len(idx_val)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'r2_train: {:.4f}'.format(r2_train),
          'l0_norm_att_train: {:.4f}'.format(l0_norm_att_train),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'r2_val: {:.4f}'.format(r2_val),
          'l0_norm_att_val: {:.4f}'.format(l0_norm_att_val),       
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()

def compute_test():
    model.eval()
    output, attention = model(features, adj)

    output = output.reshape(-1)
    loss_test = loss_fn(output[idx_test], labels[idx_test], attention, 0.01)
    r2_test = sklearn.metrics.r2_score(labels[idx_test].cpu(), output[idx_test].cpu().detach().numpy())
    l0_norm_att_test = torch.norm(attention[idx_test], 0) / len(idx_test)

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "r2_test= {:.4f}".format(r2_test),
          "l0_norm_att_test: {}".format(l0_norm_att_test))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
