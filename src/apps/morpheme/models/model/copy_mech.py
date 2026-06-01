import torch
import torch.nn as nn

class CopyMechanism(nn.Module):
    def __init__(self, input_dim, max_length):
        super().__init__()
        self.linear = nn.Linear(input_dim, max_length)

    def forward(self, x, attention_weights):
        """
        x: (batch_size, input_dim)
        attention_weights: (batch_size, max_length)
        """
        copy_probs = torch.softmax(self.linear(x), dim=-1)
        copy_weights = attention_weights * copy_probs
        return copy_weights