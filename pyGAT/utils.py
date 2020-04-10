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

def load_joshi_data():
    dataset = datasets.GeneDataset(name="Week_8_LN",
                   expr_path='../data/datastore/week8_ln_magic_expr.csv')
    dataset.load_data()
    label_df = dataset.df

    G = StringDBGraph(graph_type='coexpression')
    G = G.nx_graph.subgraph(dataset.df.columns)
    dataset.df.drop(columns=[x for x  in dataset.df.columns if x not in G.nodes()], inplace=True)

    adj = nx.adjacency_matrix(G, nodelist=dataset.df.columns)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    gene_symbol = mouse_ensg_to_symbol(datastore="../data") # ensembl to symbol
    ensembl = dict((v,k) for k,v in gene_symbol.items()) # symbol to ensembl

    features = np.array(dataset.df)
    labels = np.array(label_df[ensembl['Cd8a']])

    indices = list(range(labels.shape[0]))
    np.random.shuffle(indices)
    idx_train = indices[:5000]
    idx_val = indices[5000:8000]
    idx_test = indices[8000:]
    
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

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
