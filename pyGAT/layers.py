import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, ncells, ngenes, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.ncells = ncells
        self.ngenes = ngenes
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros
                              (size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(ncells, 2*out_features)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # self.Z = nn.Parameter(torch.zeros(size=(ncells,1,ngenes)))
        # nn.init.normal_(self.Z, mean=0, std=1)

    def forward(self, input, adj):
    
        hW = torch.einsum("cnf,fg->cng", input, self.W)
        
        """
        a_input: Whi x Whj
        a_input = torch.cat([hW.repeat(1, N).view(N * N, -1), hW.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        """
        a_input = torch.cat([hW.repeat(1,1,self.ngenes).view(self.ncells, self.ngenes * self.ngenes, self.out_features), hW.repeat(1,self.ngenes,1)], dim=1).view(self.ncells, self.ngenes, self.ngenes, 2 * self.out_features)
        
        e_input = torch.einsum("ab,acdb->acd", (self.a, a_input))
        e_input = e_input[:, -1, :] # importance of all genes to last (Cd8a) 
        
        e = self.leakyrelu(e_input).view(self.ncells, 1, self.ngenes)
        
        zero_vec = -9e15*torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec) # keep all but attention on self
        
        attention = F.dropout(attention, self.dropout, training=self.training)
        # attention = torch.where(self.Z > 0.5, attention, zero_vec)
        
        attention = F.softmax(attention, dim=2)

        h_prime = torch.bmm(attention.view(self.ncells, 1, self.ngenes), hW.view(self.ncells, self.ngenes, self.out_features))
       
        return (h_prime, attention)
        """
        if self.concat:
             return F.elu(h_prime)
        else:
             return h_prime
       """
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return (h_prime)
        """
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime
        """
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
