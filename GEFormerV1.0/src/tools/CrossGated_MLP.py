import torch
import torch.nn as nn

class CrossGatedMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        # MLPs for computing x1 and x2 hidden representations
        self.mlp_x1 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, input_size),
            nn.GELU()
        )
        
        self.mlp_x2 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, input_size),
            nn.GELU()
        )
        
        # MLPs for computing x1 and x2 gates
        self.gate_x1 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        
        self.gate_x2 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # Compute x1 and x2 hidden representations
        hidden_x1 = self.mlp_x1(x1)
        hidden_x2 = self.mlp_x2(x2)
        
        # Compute x1 and x2 gates
        gate_x1 = self.gate_x1(x1)
        gate_x2 = self.gate_x2(x2)
        
        # Compute fused features using the cross-gated MLP mechanism
        cross_gated_x1 = (1 - gate_x1) * hidden_x1 + gate_x2 * hidden_x2
        cross_gated_x2 = (1 - gate_x2) * hidden_x2 + gate_x1 * hidden_x1
        
        # Concatenate the fused features and return
        fused_features = torch.cat([cross_gated_x1, cross_gated_x2], dim=1)
        return fused_features









