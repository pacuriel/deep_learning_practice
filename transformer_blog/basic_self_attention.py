"""From blog post by Peter Bloem: https://peterbloem.nl/blog/transformers."""
import torch
import torch.nn.functional as F

# Set dimensions of input sequence
batch_size_b = 2 # Number of sequences in a batch
seq_length_t = 4 # Length of the input sequence (number of words/tokens)
embedding_dim_k = 8 # Dimensionality of the embedding vector for each word/token

# Create random input sequence
x = torch.randn(batch_size_b, seq_length_t, embedding_dim_k)

# Set of raw dot products w_ij' (X multiplied by X^T)
raw_weights = torch.bmm(x, x.transpose(1, 2))

# Convert raw weights to probabilities using (row-wise) softmax
weights = F.softmax(raw_weights, dim=2) 

# Compute outputs sequence (rows are weighted sums over the rows of x)
y = torch.bmm(weights, x)