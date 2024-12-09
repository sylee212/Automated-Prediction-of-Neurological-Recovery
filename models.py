import torch
from torch.nn import Linear as Lin
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn 
from torch_geometric.nn import global_mean_pool,global_add_pool, global_max_pool 

class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GCNLayer, self).__init__()
        self.include_bias = bias
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.weight)
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj, edge_w=None):
        epsilon=1e-10
        support = torch.matmul(x, self.weight)
        
        # If edge_w is not provided, assume binary connections
        if edge_w is None:
            edge_w = torch.ones(adj.size(1)).to(x.device)
            
        # Convert indices and edge weights to a dense adjacency matrix
        adj_size = (x.size(0), x.size(0))
        adj_dense = torch.sparse_coo_tensor(adj, edge_w, size=adj_size).to_dense()
        
        # Add self-connections (A_hat)
        # Create an identity matrix with the same dtype as the adjacency matrix
        identity_matrix = torch.eye(x.size(0), dtype=adj_dense.dtype, device=adj_dense.device)
        # Add the identity matrix to the adjacency matrix
        adj_dense = adj_dense + identity_matrix

        # normalize adj
        # Calculate the degree matrix (sum along rows)
        degree_matrix = torch.sum(adj_dense, dim=1, keepdim=True)
        # Calculate the inverse square root of the degree matrix
        degree_matrix_inv_sqrt = 1. / torch.sqrt(degree_matrix + epsilon)
        # Create a diagonal matrix with the inverse square root of degrees
        D_inv_sqrt = torch.diag_embed(degree_matrix_inv_sqrt.squeeze())
        # Symmetric normalization: D_inv_sqrt * A * D_inv_sqrt
        normalized_adjacency = torch.matmul(torch.matmul(D_inv_sqrt, adj_dense), D_inv_sqrt)

        # Multiply the dense adjacency matrix with the support tensor
        if self.include_bias:
            output = torch.matmul(normalized_adjacency, support) + self.bias
        else:
            output = torch.matmul(adj_dense, support)
        return output
    
    
class GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, hgc, 
                 lg, nrois=20, aggr_method='mean'):
        super(GCN, self).__init__()
        hidden = [hgc for i in range(lg)]
        self.dropout = dropout
        self.aggr_type = aggr_method
        bias = False 
        self.relu = torch.nn.ReLU(inplace=True) 
        self.lg = lg 
        self.gconv = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i==0  else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], 2, normalization='sym', bias=bias)) 
            # self.gconv.append(tg.nn.GCNConv(in_channels, hidden[i], bias=bias))
            # self.gconv.append(GCNLayer(in_channels, hidden[i], bias=bias))
            self.batch_norms.append(tg.nn.norm.BatchNorm(hidden[i]))
        cls_input_dim = sum(hidden) 
        
        if self.aggr_type == 'flatten':
            cls_input_dim = cls_input_dim * nrois
        self.cls = nn.Sequential(
                torch.nn.Linear(cls_input_dim, 256),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(256), 
                torch.nn.Linear(256, num_classes))
            
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edgenet_input, batch, enforce_edropout=False): 
        features = F.dropout(features, self.dropout, self.training)
        h = self.gconv[0](features, edge_index)
        
        h = self.batch_norms[0](h)
        h = self.relu(h)
        h = F.dropout(h, self.dropout, self.training)
        h0 = h
        for conv, batch_norm in zip(self.gconv[1:], self.batch_norms[1:]):
            h = conv(h, edge_index)
            # h = batch_norm(h)
            h = self.relu(h)
            h = batch_norm(h)
            h = F.dropout(h, self.dropout, self.training)
            jk = torch.cat((h0, h), axis=1)
            h0 = jk
            
        batch_size = len(torch.unique(batch))
        
        # 2. perform graph-level classification
        if self.aggr_type == 'mean':
            # average aggregate across nodes (batch,z) batch=number subjects
            # z_agg = x.view(batch_size,-1,x.shape[-1]).mean(1)  # average aggregation
            z_agg = global_mean_pool(jk, batch)  # [batch_size, hidden_channels]
            # z_agg = segment_csr(jk, batch, reduce="mean")
        elif self.aggr_type == 'flatten':
            z_agg = jk.view(batch_size,-1,jk.shape[-1]).flatten(1)
        elif self.aggr_type == 'sum':
            z_agg = global_add_pool(jk, batch)
        elif self.aggr_type == 'max':
            z_agg = global_max_pool(jk, batch) 
        
        logit = self.cls(z_agg)

        return z_agg, logit
