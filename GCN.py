import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.Conv1 = GCNConv(input_dim, hidden_dim)
        self.Conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self,x,edge_index,batch):
        x = self.Conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.Conv2(x, edge_index)
        F.log_softmax(x, dim=1)
        return x
    

