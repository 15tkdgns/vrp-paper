import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGraphVol(nn.Module):
    """
    Adaptive Graph Model for Volatility Prediction.
    Learns dynamic correlations between assets to capture spillover effects.
    """
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels=1):
        super(AdaptiveGraphVol, self).__init__()
        self.num_nodes = num_nodes
        
        # 1. Node Embeddings for Adaptive Adjacency
        self.e1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.e2 = nn.Parameter(torch.randn(num_nodes, 10))
        
        # 2. Linear Transformation for Node Features
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        
        # 3. Simple Graph Conv (A * X * W)
        self.gcn_w = nn.Parameter(torch.randn(hidden_channels, hidden_channels))
        
        # 4. Readout Head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x):
        """
        x: (Batch, NumNodes, InChannels) 
        InChannels = SeqLen * 2 (LogRet, LogRV)
        """
        B, N, C = x.shape
        
        # 1. Generate Adaptive Adjacency Matrix
        # A: (N, N)
        adp = F.softmax(F.relu(torch.mm(self.e1, self.e2.t())), dim=-1)
        
        # 2. Feature Projection
        h = self.lin_in(x) # (B, N, H)
        
        # 3. Message Passing (Graph Convolution)
        # h: (B, N, H), adp: (N, N)
        # result: (B, N, H)
        h_graph = torch.matmul(adp, h) # Mix features based on learned graph
        h_graph = torch.matmul(h_graph, self.gcn_w)
        h_graph = F.relu(h_graph)
        
        # 4. Output
        # We need to predict Target LogRV for each node
        out = self.head(h_graph) # (B, N, 1)
        return out.squeeze(-1) # (B, N)
