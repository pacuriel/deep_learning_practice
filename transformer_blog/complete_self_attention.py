"""From blog post by Peter Bloem: https://peterbloem.nl/blog/transformers."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):
        super().__init__() # Init. parent class (nn.Module)

        # Ensure the embedding dimension is divisible by the number of heads
        assert k % heads == 0 

        self.k, self.heads = k, heads
        self.mask = mask

        # Setting up linear transformations (nn.Linear with bias set to False)
        self.to_keys = nn.Linear(k, k, bias=False)
        self.to_queries = nn.Linear(k, k, bias=False)
        self.to_values = nn.Linear(k, k, bias=False)

        # Output linear transformation (applied after multi-head self-attention)
        self.unify_heads = nn.Linear(k, k)

    def forward(self, x):
        # Batch size, sequence length, embedding dimension
        b, t, k = x.size() 

        h = self.heads # Number of attention heads (not required to set here)

        # Compute keys, queries, values
        queries = self.to_queries(x)
        keys = self.to_keys(x)
        values = self.to_values(x)

        s = k // h # Dimension of each attention head

        # Adding a dimenison that iterates over each attention head
        # Example: for single vector of dimension k, reshape into a matrix of shape h x (k // h)
        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # Folding the heads into the batch dimension for efficient computation (torch.bmm)
        # Note: transpose so head and batch dimension are next to each other
        # Note: able to use reshape() instead of view() to avoid call to contiguous()
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # Computing dot products (containing raw weights or attention scores before softmax)
        dot = torch.bmm(queries, keys.transpose(1, 2)) # size = (b * h, t, t)

        # Scale the dot products by the square root of the dimension of each head
        dot = dot / (s ** (1 / 2))

        # Normalize using softmax (dim = 2 to normalize across the sequence length dimension)
        dot = F.softmax(dot, dim=2) # Row-wise normalized weights

        # Apply the self-attention weights to the values (output for each attention head)
        out = torch.bmm(dot, values).view(b, h, t, s)

        # Transpose again so head and embedding dimensions are next to each other
        # Reshape to get concatenated vectors of dimension k
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        
        # Pass through final linear/projection layer
        return self.unify_heads(out)