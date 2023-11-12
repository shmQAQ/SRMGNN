# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import math
import torch
from torch import nn, optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
from SMGNNmodels import GraphTransformer
from data import MoleculeDataset
from utils import count_parameters, smile_to_graph
from position_encoding import POSENCODINGS
from gnn_layers import GNN_TYPES
from timeit import default_timer as timer


def load_args():
    parser = argparse.ArgumentParser(
        description='Structure-Aware Transformer on SBM datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--task_name', type=str, default="emi FWHM (cm-1)",
                        help='name of tasks')
    parser.add_argument('--num-heads', type=int,
                        default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int,
                        default=6, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=32,
                        help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--abs-pe', type=str, default=None, choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=3,
                        help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=5000,
                        help="number of iterations for warmup")
    parser.add_argument('--layer-norm', action='store_true',
                        help='use layer norm instead of batch norm')
    parser.add_argument('--gnn-type', type=str, default='pna',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=3,
                        help="number of layers for GNNs")
    parser.add_argument('--weight-class', action='store_true',
                        help='weight classes or not')

    parser.add_argument('--se', type=str, default="khopgnn",
                        help='Extractor type: khopgnn, or gnn')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/seed{}'.format(args.seed)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.sparse:
            outdir = outdir + '/sparse'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        pedir = 'None' if args.abs_pe is None else '{}_{}'.format(
            args.abs_pe, args.abs_pe_dim)
        outdir = outdir + '/{}'.format(pedir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        if args.se == "khopgnn":
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        else:
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0
    n_sample = 0

    tic = timer()
    for i, data in enumerate(loader):
        size = len(data.y)
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.abs_pe == 'lap':
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(data.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.abs_pe = data.abs_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)
        print(output.shape,data.y.shape)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * size
        n_sample += size

    toc = timer()
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False, split='Val'):
    model.eval()

    running_loss = 0.0
    tic = timer()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if use_cuda:
                data = data.cuda()

            output = model(data)
            loss = criterion(output, data.y)

            running_loss += loss.item() 
    toc = timer()

    epoch_loss = running_loss 
    print('{} loss: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, toc - tic))
    return epoch_loss


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    n_tags = 8 # number of features in data.x
    num_class = 1 # do regression
    args.num_class = num_class
    num_edge_features = 0
    task_name = args.task_name
    # create cache path
    cache_path = '../cache/{}_{}hop/'.format(task_name, args.k_hop)
    if args.se == 'khopgnn':
        os.makedirs(cache_path, exist_ok=True)

    df = pd.read_csv('data/raw/data.csv', usecols=['smiles', task_name])
    graphs = []
    for i in range(len(df)):
        task = df.iloc[i][task_name]
        if math.isnan(task):
            continue
        graph = smile_to_graph(df.iloc[i]['smiles'])
        graph.y = torch.tensor(task)
        graphs.append(graph)

    dataset = MoleculeDataset(graphs, degree=True, k_hop=3, se="khopgnn", use_subgraph_edge_attr=True,
                              cache_path=None, return_complete_index=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_dset, val_dset, test_dset = random_split(dataset=dataset, lengths=[0.8,0.1,0.1], generator=generator)
    input_size = n_tags
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)
    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dset)
            abs_pe_encoder.apply_to(val_dset)

    deg = torch.cat([
        utils.degree(train_dset[i].edge_index[1],
                     num_nodes=train_dset[i].num_nodes)
        for i in range(len(train_dset))])

    model = GraphTransformer(in_size=input_size,
                             num_class=num_class,
                             d_model=args.dim_hidden,
                             dim_feedforward=2*args.dim_hidden,
                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             gnn_type=args.gnn_type,
                             k_hop=args.k_hop,
                             use_edge_attr=False,
                             num_edge_features=num_edge_features,
                             deg=deg,
                             in_embed=False,
                             se=args.se,
                             use_global_pool=False)
    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.5,
                                                            patience=15,
                                                            min_lr=1e-05,
                                                            verbose=False)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5

        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    if abs_pe_encoder is not None:
        abs_pe_encoder.apply_to(test_dset)

    print("Training...")
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {}".format(epoch + 1,
              args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
        val_loss = eval_epoch(
            model, val_loader, criterion, args.use_cuda, split='Val')
        print(train_loss, val_loss)

    print("Testing...")
    test_loss = eval_epoch(
        model, test_loader, criterion, args.use_cuda, split='Test')
    print("Test loss: {}".format(test_loss))


if __name__ == "__main__":
    main()
