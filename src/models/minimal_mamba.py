import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MinimalMambaBlock(nn.Module):
    """
    A lightweight, pure-PyTorch implementation of the Mamba Block (S6).
    Simplified for portability and ease of use in non-CUDA environments (like WSL without nvcc).
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        
        # 1. Input Projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2. 1D Convolution (Causal)
        # Groups = d_inner for depthwise
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # 3. State Space Model Parameters
        # x_proj maps input -> (Delta, B, C)
        # Delta: (B, L, d_inner)
        # B: (B, L, d_state)
        # C: (B, L, d_state)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state * 2, bias=False)
        
        # dt_proj maps Delta -> (B, L, d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # A: (d_inner, d_state)
        A_init = torch.arange(1, d_state + 1).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init.float()))
        
        # D: (d_inner)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 4. Output Projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        x: (Batch, SeqLen, Dim)
        """
        B, L, D = x.shape
        
        # 1. Project
        xz = self.in_proj(x) # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1) # (B, L, d_inner)
        
        # 2. Conv1d
        # Rearrange for Conv: (B, Dim, Seq)
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L] # Trim padding
        x_conv = self.act(x_conv).transpose(1, 2) # (B, L, d_inner)
        
        # 3. SSM
        # Calculate Delta, B, C dependent on Input (Selective!)
        x_dbl = self.x_proj(x_conv) # (B, L, d_inner + 2*d_state)
        delta, B_ssm, C_ssm = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        
        # Softplus for positive delta
        delta = F.softplus(self.dt_proj(delta)) # (B, L, d_inner)
        
        # Discretize A
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        # Run SSM Scan (Sequential for simplicity in PyTorch)
        # y_t = SSM(x_conv, delta, A, B, C)
        y = self.selective_scan(x_conv, delta, A, B_ssm, C_ssm, self.D)
        
        # 4. Gating & Output
        y = y * self.act(z)
        out = self.out_proj(y)
        
        return out
    
    def selective_scan(self, u, delta, A, B, C, D):
        """
        Naive sequential implementation of Selective Scan.
        u: (Batch, L, d_inner)
        delta: (Batch, L, d_inner)
        A: (d_inner, d_state)
        B: (Batch, L, d_state)
        C: (Batch, L, d_state)
        D: (d_inner)
        """
        Batch, L, d_inner = u.shape
        d_state = A.shape[1]
        
        # Initialize state
        h = torch.zeros(Batch, d_inner, d_state, device=u.device)
        
        ys = []
        
        # Discretize A once? No, A_bar depends on delta (time-varying)
        # exp(A * delta)
        
        # Sequential Loop (Slow but functional)
        # Parallel scan is possible but requires complex code.
        for t in range(L):
            # 1. Discretize Parameters for step t
            dt = delta[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            dA = torch.exp(A * dt) # (B, d_inner, d_state)
            dB = B[:, t, :].unsqueeze(1) * dt # (B, d_inner, d_state)
            
            # 2. Update State
            # h[t] = dA * h[t-1] + dB * u[t]
            u_t = u[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            h = dA * h + dB * u_t # (B, d_inner, d_state)
            
            # 3. Output
            # y[t] = C[t] * h[t] + D * u[t]
            C_t = C[:, t, :].unsqueeze(1) # (B, 1, d_state)
            y_t = torch.sum(h * C_t, dim=-1) # (B, d_inner)
            
            # Add Residual D
            y_t = y_t + u[:, t, :] * D
            
            ys.append(y_t)
            
        return torch.stack(ys, dim=1) # (B, L, d_inner)
