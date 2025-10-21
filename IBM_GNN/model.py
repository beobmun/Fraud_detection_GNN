import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax

class IntraMetapathAttention(MessagePassing):
    def __init__(self, in_channels_src, in_channels_dest, out_channels, edge_dim, heads, dropout=0.2):
        super().__init__(aggr='add', node_dim=-2)
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)

        self.lin_src = Linear(in_channels_src, heads * out_channels)
        self.lin_dest = Linear(in_channels_dest, heads * out_channels)
        if edge_dim > 0:
            self.lin_edge = Linear(edge_dim, heads * out_channels)
        else:
            self.lin_edge = nn.Identity()

        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        nn.init.xavier_uniform_(self.att)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads, self.out_channels)
        return aggr_out.mean(dim=1)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # x_i: target node attributes, x_j: source node attributes
        h_i = self.lin_dest(x_i).view(-1, self.heads, self.out_channels)
        h_j = self.lin_src(x_j).view(-1, self.heads, self.out_channels)
        if isinstance(self.lin_edge, nn.Identity):
            e_ij = torch.zeros_like(h_j)
        else:
            e_ij = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)

        alpha_src = (h_j * self.att).sum(dim=-1)
        alpha_dest = (h_i * self.att).sum(dim=-1)
        alpha_edge = (e_ij * self.att).sum(dim=-1)
        alpha = alpha_src + alpha_dest + alpha_edge
        alpha = F.leaky_relu(alpha, 0.2)
        
        alpha_flat = alpha.view(-1)
        index_repeated = index.repeat_interleave(self.heads)
        alpha_softmaxed_flat = softmax(alpha_flat, index_repeated, ptr, size_i * self.heads)
        alpha_final = alpha_softmaxed_flat.view_as(alpha).contiguous()
        alpha_final = self.dropout(alpha_final)

        final_message = torch.mul(h_j, alpha_final.unsqueeze(-1))
        return final_message.view(-1, self.heads * self.out_channels)
    
class GNNLayer(nn.Module):
    def __init__(self, in_channels_dict, out_channels, metadata, heads):
        super().__init__()
        self.out_channels = out_channels
        self.node_types = metadata[0]
        self.intra_attn_dict = nn.ModuleDict()
        for edge_type in metadata[1]:
            src_type, _, dest_type = edge_type
            in_channels_src = in_channels_dict[src_type]
            in_channels_dest = in_channels_dict[dest_type]
            edge_dim = in_channels_dict.get(edge_type, 0)

            edge_type_str = "_".join(edge_type)
            self.intra_attn_dict[edge_type_str] = IntraMetapathAttention(in_channels_src, in_channels_dest, out_channels, edge_dim, heads)
        
        self.inter_attn_mlp = nn.Sequential(
            Linear(out_channels, 64),
            nn.Tanh(),
            Linear(64, 1, bias=False)
        )

        self.lin_final = nn.ModuleDict({
            node_type: Linear(in_channels_dict[node_type], out_channels) for node_type in self.node_types
        })

    def forward(self, h_dict, edge_index_dict, edge_attr_dict):
        metapath_outs = {node_type: [] for node_type in h_dict.keys()}

        for edge_type, edge_index in edge_index_dict.items():
            src, _, dest = edge_type
            if edge_index.size(1) == 0:
                continue
            edge_attr = edge_attr_dict.get(edge_type, torch.empty((edge_index.size(1), 0), device=edge_index.device))
            edge_type_str = "_".join(edge_type)

            out = self.intra_attn_dict[edge_type_str]((h_dict[src], h_dict[dest]), edge_index, edge_attr)
            metapath_outs[dest].append(out)

        final_h_dict = dict()
        for node_type in self.node_types:
            outs = metapath_outs.get(node_type, [])
            final_h = self.lin_final[node_type](h_dict[node_type])
            if not outs:
                final_h_dict[node_type] = final_h
                continue
            stacked_outs = torch.stack(outs, dim=1)
            num_nodes, num_metapaths, _ = stacked_outs.shape
            attn_input = stacked_outs.view(-1, self.out_channels)
            raw_attn_scores = self.inter_attn_mlp(attn_input)
            attn_scores = raw_attn_scores.view(num_nodes, num_metapaths)
            attn_weights = F.softmax(attn_scores, dim=1)
            aggr_out = torch.sum(stacked_outs * attn_weights.unsqueeze(-1), dim=1)

            final_h_dict[node_type] = final_h + aggr_out

        return final_h_dict

class TemporalEncoder(nn.Module):
    def __init__(self, node_features_dim, edge_features_dim, zip_emb_dim, mcc_emb_dim, gnn_hidden_channels, gru_hidden_channels, metadata, num_zip_idx, num_mcc_idx, heads):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        self.node_zip_embedding = nn.Embedding(num_zip_idx, zip_emb_dim)
        self.edge_mcc_embedding = nn.Embedding(num_mcc_idx, mcc_emb_dim)
        self.edge_zip_embedding = nn.Embedding(num_zip_idx, zip_emb_dim)

        self.node_lin_dict = nn.ModuleDict()
        self.edge_lin_dict = nn.ModuleDict()

        gnn_in_channels_dict = dict()
        for node_type in self.node_types:
            in_dim = node_features_dim[node_type]
            if node_type in ['user', 'merchant']:
                in_dim += zip_emb_dim
            self.node_lin_dict[node_type] = Linear(in_dim, gnn_hidden_channels)
            gnn_in_channels_dict[node_type] = gnn_hidden_channels

        for edge_type in self.edge_types:
            in_dim = edge_features_dim.get(edge_type, 0)
            if edge_type in [('card', 'transaction', 'merchant'), ('merchant', 'transaction_by', 'card'), ('merchant', 'refund', 'card'), ('card', 'refund_by', 'merchant')]:
                in_dim += mcc_emb_dim + zip_emb_dim

            edge_type_str = "_".join(edge_type)
            if in_dim > 0:
                self.edge_lin_dict[edge_type_str] = Linear(in_dim, gnn_hidden_channels)
                gnn_in_channels_dict[edge_type] = gnn_hidden_channels
            else:
                self.edge_lin_dict[edge_type_str] = nn.Identity()
                gnn_in_channels_dict[edge_type] = 0

        self.gnn = GNNLayer(gnn_in_channels_dict, gnn_hidden_channels, metadata, heads)

        self.gru_dict = nn.ModuleDict({
            node_type: nn.GRU(gnn_hidden_channels, gru_hidden_channels) for node_type in self.node_types
        })

    def forward(self, snapshot_seq):
        h_gru_dict = {node_type: None for node_type in self.node_types}
        
        for snapshot in snapshot_seq:
            h_dict = dict()
            for node_type in self.node_types:
                if node_type in ['user', 'merchant']:
                    node_zip_embs = self.node_zip_embedding(snapshot[node_type].zip_idx)
                    combined_features = torch.cat([snapshot[node_type].x, node_zip_embs], dim=-1)
                else:
                    combined_features = snapshot[node_type].x
                h_dict[node_type] = F.leaky_relu(self.node_lin_dict[node_type](combined_features))

            e_dict = dict()
            for edge_type in self.edge_types:
                edge_type_str = "_".join(edge_type)
                if edge_type in [('card', 'transaction', 'merchant'), ('merchant', 'transaction_by', 'card'), ('merchant', 'refund', 'card'), ('card', 'refund_by', 'merchant')]:
                    edge_mcc_embs = self.edge_mcc_embedding(snapshot[edge_type].edge_mcc_idx)
                    edge_zip_embs = self.edge_zip_embedding(snapshot[edge_type].edge_zip_idx)
                    combined_edge_features = torch.cat([snapshot[edge_type].edge_attr, edge_mcc_embs, edge_zip_embs], dim=-1)
                    e_dict[edge_type] = F.leaky_relu(self.edge_lin_dict[edge_type_str](combined_edge_features))
                else:
                    e_dict[edge_type] = torch.empty((snapshot[edge_type].edge_index.size(1), 0), device=snapshot[edge_type].edge_index.device)
                
            node_embs_dict = self.gnn(h_dict, snapshot.edge_index_dict, e_dict)

            for node_type in self.node_types:
                if node_embs_dict[node_type].size(0) > 0:
                    gru_input = node_embs_dict[node_type].unsqueeze(0)
                    _, h_gru_dict[node_type] = self.gru_dict[node_type](gru_input, h_gru_dict[node_type])

        final_embs_dict = {node_type: h.squeeze(0) for node_type, h in h_gru_dict.items() if h is not None}
        return final_embs_dict
    
class EdgeDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.mlp = nn.Sequential(
            Linear(2 * in_channels, in_channels),
            nn.ReLU(),
            Linear(in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embs_dict, edge_index_dict):
        predictions = dict()
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dest_type = edge_type
            src_embs = node_embs_dict[src_type][edge_index[0]]
            dest_embs = node_embs_dict[dest_type][edge_index[1]]
            combined_embs = torch.cat([src_embs, dest_embs], dim=-1)
            predictions[edge_type] = self.mlp(combined_embs)

        return predictions

class FraudDetectionModel(nn.Module):
    def __init__(self, node_features_dim, edge_features_dim, zip_emb_dim, mcc_emb_dim, gnn_hidden_channels, gru_hidden_channels, metadata, num_zip_idx, num_mcc_idx, heads):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(node_features_dim, edge_features_dim, zip_emb_dim, mcc_emb_dim, gnn_hidden_channels, gru_hidden_channels, metadata, num_zip_idx, num_mcc_idx, heads)
        self.edge_decoder = EdgeDecoder(gru_hidden_channels)
        self.node_types = metadata[0]

    def forward(self, history_snapshot_batch, target_batch):
        batch_final_node_embs_dict_list = list()
        # device = next(self.parameters()).device
        device = target_batch.x_dict[self.node_types[0]].device
        for history_snapshots in history_snapshot_batch:
            final_embs_dict = self.temporal_encoder(history_snapshots)
            batch_final_node_embs_dict_list.append(final_embs_dict)
            final_node_embs_across_batches = {node_type: [] for node_type in self.node_types}
            for node_type in self.node_types:
                for batch_dict in batch_final_node_embs_dict_list:
                    if node_type in batch_dict and batch_dict[node_type].size(0) > 0:
                        final_node_embs_across_batches[node_type].append(batch_dict[node_type])
                
                if final_node_embs_across_batches[node_type]:
                    final_node_embs_across_batches[node_type] = torch.cat(final_node_embs_across_batches[node_type], dim=0)
                else:
                    final_node_embs_across_batches[node_type] = torch.empty((0, self.edge_decoder.in_channels), device=device)

        predictions = self.edge_decoder(final_node_embs_across_batches, target_batch.edge_index_dict)
        return predictions
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1 - pt) ** self.gamma
        focal_loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss