#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
from torch_geometric.data import Dataset, download_url, Data
from scipy.io import loadmat
import torch
import torch.utils.data as tud


class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None,
                edge_weight_s=None, edge_weight_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_weight_s = edge_weight_s
        self.edge_weight_t = edge_weight_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

        
class ABCDDataset(Dataset):
    def __init__(self, root, size=8710, if_normalize=True):
        super().__init__(root)
#         self.processed_dir = root 
        self.size = size
        self.if_normalize = if_normalize

    @property
    def processed_file_names(self):
        return [str(fileidx)+'.mat' for fileidx in range(self.size)]

    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self, idx):
        data = loadmat(osp.join(self.root, 'ABCD'+str(idx+1)+'.mat'))
        fc, sc = torch.tensor(data['fc']), torch.tensor(data['sc'])
        cont = [[],[]]
        
        # build graph with connectivity matrix
        for cidx,cm in enumerate([fc, sc]):
            if self.if_normalize:
                cm = (cm - torch.min(cm)) / (torch.max(cm)-torch.min(cm))
            # threshold
            absfc = abs(cm)
            #cm[absfc<torch.mean(absfc)-torch.std(absfc)] = 0

            edge_index = torch.zeros([268*267,2],dtype=torch.int64)
            edge_weight = torch.zeros(268*267)
            true_idx = 0
            x = torch.eye(cm.shape[-1])

            for i in range(0, cm.shape[-1]):
                for j in range(i+1,cm.shape[-1]):
                    edge_index[true_idx] = torch.tensor([i,j], dtype=torch.int64)
                    edge_index[true_idx+1] = torch.tensor([j,i], dtype=torch.int64)
                    edge_weight[true_idx] = cm[i][j]
                    edge_weight[true_idx+1] = cm[i][j]
                    true_idx += 2
            cont[cidx] = [x,edge_index,edge_weight]

#             data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_weight=edge_weight)
        data = PairData(x_s=cont[0][0], edge_index_s=cont[0][1].t().contiguous(), edge_weight_s=cont[0][2],
                            x_t=cont[1][0], edge_index_t=cont[1][1].t().contiguous(), edge_weight_t=cont[1][2])

        return data
    
    
################################################
# load matrix version
class ABCDCNNset(tud.Dataset):
    def __init__(self, root, size=8710):
        self.root = root
        self.size = size


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        data = loadmat(osp.join(self.root, 'ABCD'+str(idx+1)+'.mat'))
        fc, sc = torch.tensor(data['fc']), torch.tensor(data['sc'])
        return fc, sc, 
    
    
    

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_geometric.nn as gnn

class weiGCNConv(MessagePassing):
    # weighted GCN convolution
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0), edge_attr=edge_weight)
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)

    def message(self, x_j, norm, edge_weight):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * edge_weight.view(-1, 1) * x_j