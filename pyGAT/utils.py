import numpy as np
import scipy.sparse as sp
import torch
import os
import pickle
from gtfparse import read_gtf

import sys
sys.path += ['..']
import networkx as nx
from data import datasets
from data.gene_graphs import StringDBGraph
from data.utils import record_result, mouse_ensg_to_symbol

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_synthetic_data(gene):
    dataset = datasets.SyntheticDataset(name='CD8_count_synthetic', expr_path='../../project/CD8_T_count.csv',label_path='../../project/CD8_T_label.csv')
    dataset.load_data()

    gene_symbol = mouse_ensg_to_symbol(datastore="../data") # ensembl to symbol
    ensembl = dict((v,k) for k,v in gene_symbol.items()) # symbol to ensembl
    cols = set(dataset.df.columns).intersection(set(ensembl.keys()))
    
    G = StringDBGraph(graph_type='all')
    G = G.nx_graph.subgraph([ensembl[x] for x in cols])
    ngenes = list(G.neighbors(ensembl[gene]))
    ngenes.append(ensembl[gene])
    G = G.subgraph(ngenes)

    # save memory
    dataset.df = dataset.df[cols].drop(columns=[x for x in cols if x not in [gene_symbol[y] for y in G.nodes()]])
    dataset.df = dataset.df[[gene_symbol[x] for x in list(ngenes)]]
    adj = nx.adjacency_matrix(G, nodelist=ngenes)

    features = dataset.df # samples x features
    labels = np.array(dataset.labels).reshape(-1) # get for one gene

    indices = list(range(labels.shape[0]))
    np.random.shuffle(indices)
    idx_train = indices[:500]
    idx_val = indices[500:700]
    idx_test = indices[700:]
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    # for each cell, repeat same adjacency vector (row: mygene, column: first-degree neighbors of mygene)
    adj = np.tile(np.array(adj.todense()[-1,:]), features.shape[0])
    adj = torch.FloatTensor(adj).view(features.shape[0], 1, features.shape[1]) 

    features = torch.FloatTensor(np.array(features)).view(features.shape[0], features.shape[1], 1)
    labels = torch.FloatTensor(labels).view(labels.shape[0])

    return adj, features, labels, idx_train, idx_val, idx_test, dataset.df.columns

def load_joshi_data(gene):
    dataset = datasets.GeneDataset(name="Week_8_LN",
                   expr_path='../data/datastore/week8_ln_raw_expr.csv')
    
    label_dataset = datasets.GeneDataset(name="Week_8_LN",
                   expr_path='../data/datastore/week8_ln_magic_expr.csv')
    
    dataset.load_data()
    label_dataset.load_data()
    gene_symbol = mouse_ensg_to_symbol(datastore="../data") # ensembl to symbol
    ensembl = dict((v,k) for k,v in gene_symbol.items()) # symbol to ensembl

    # G = StringDBGraph(graph_type='coexpression')
    G = StringDBGraph(graph_type='all')
    G = G.nx_graph.subgraph(dataset.df.columns)
    ngenes = list(G.neighbors(ensembl[gene]))
    ngenes.append(ensembl[gene])
    G = G.subgraph(ngenes)
    
    # save memory
    dataset.df.drop(columns=[x for x in dataset.df.columns if x not in G.nodes()], inplace=True)
    dataset.df = dataset.df[list(ngenes)]
    adj = nx.adjacency_matrix(G, nodelist=ngenes)

    features = dataset.df # samples x features
    labels = np.array(label_dataset.df[ensembl[gene]]).reshape(-1) # get for one gene

    indices = list(range(labels.shape[0]))
    np.random.shuffle(indices)
    idx_train = indices[:800]
    idx_val = indices[800:1200]
    idx_test = indices[1200:]
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    # for each cell, repeat same adjacency vector (row: mygene, column: first-degree neighbors of mygene)
    adj = np.tile(np.array(adj.todense()[-1,:]), features.shape[0])
    adj = torch.FloatTensor(adj).view(features.shape[0], 1, features.shape[1]) 

    features = torch.FloatTensor(np.array(features)).view(features.shape[0], features.shape[1], 1)
    labels = torch.FloatTensor(labels).view(labels.shape[0])

    return adj, features, labels, idx_train, idx_val, idx_test, dataset.df.columns
    
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# classification accuracy
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
