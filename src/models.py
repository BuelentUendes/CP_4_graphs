## Basic implementations of several base class graph models
# Code adapted from: https://github.com/soroushzargar/DAPS/blob/main/torch-conformal/gnn_cp/models/graph_models.py

# Standard PyTorch library imports
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, APPNP, SAGEConv

class GCN(nn.Module):

    def __init__(self, n_features, n_classes, **kwargs):
        """
        Standard GCN architecture
        :param n_features: number of features
        :param n_classes: number of classes
        :param kwargs: additional parameters, as specified in the models_config_file.yaml
        """

        super().__init__()
        n_hidden = kwargs.get("n_hidden", 64)
        p_dropout = kwargs.get("p_dropout", 0.8)

        self.conv1 = GCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, edge_index, edge_weight=None):

        x = self.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)

        return x

class GAT(nn.Module):

    def __init__(self, n_features, n_classes, **kwargs):
        """
        Standard Graph Attention Model
        :param n_features: number of features
        :param n_classes: number of classes
        :param kwargs: additional parameters, as specified in the models_config_file.yaml
        """

        super().__init__()
        n_hidden = kwargs.get("n_hidden", 64)
        n_heads = kwargs.get("n_heads", 8)
        p_dropout = kwargs.get("p_dropout", 0.6)
        p_dropout_attention = kwargs.get("p_dropout_attention", 0.3)

        self.conv1 = GATConv(n_features, n_hidden, heads=n_heads, dropout=p_dropout_attention, concat=True)
        self.conv2 = GATConv(n_hidden*n_heads, n_classes, heads=1, dropout=p_dropout_attention, concat=False)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, edge_index):

        x = self.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return x

class SAGE(nn.Module):

    def __init__(self, n_features, n_classes, aggregator="mean", **kwargs):

        """
        Standard GraphSAGE model
        :param n_features: number of features
        :param n_classes: number of classes
        :param aggregator: aggregate function, choose between mean, max, sum
        :param kwargs: additional parameters as specified in the models_config_file.yaml
        """

        super().__init__()
        n_hidden = kwargs.get("n_hidden", 64)
        p_dropout = kwargs.get("p_dropout", 0.4)

        self.conv1 = SAGEConv(n_features, n_hidden, normalize=True, aggr=aggregator)
        self.lin1 = nn.Linear(n_features + n_hidden, n_hidden)
        self.conv2 = SAGEConv(n_hidden, n_classes, normalize=True, aggr=aggregator)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, edge_index):

        hidden = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.lin1(torch.cat([x, hidden], dim=1)))

        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return x

class APPNPNet(nn.Module):

    def __init__(self, n_features, n_classes, **kwargs):

        """
        Implementation of the approximate personalized pagerank neural prediction layer
        :param n_features: number of features
        :param n_classes: number of classes
        :param kwargs: additional parameters as specified in the models_config_file.yaml
        """

        super().__init__()
        n_hidden = kwargs.get("n_hidden", 64)
        p_dropout = kwargs.get("p_dropout", 0.4)
        k = kwargs.get("k", 10)
        alpha = kwargs.get("alpha", 0.1)

        self.lin1 = nn.Linear(n_features, n_hidden, bias=False)
        self.lin2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.appnp = APPNP(K=k, alpha=alpha)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, edge_index):

        x = self.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.appnp(x, edge_index)

        return x

        #Check if this is correct?
        #return F.log_softmax(x, dim=1)





