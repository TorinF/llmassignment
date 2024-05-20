# add all  your Encoder and Decoder code here
#Notice that the transformer blocks (encoder, decoders, ...) have been implemented in various libraries PyTorch.
import matplotlib.pyplot as plt
import torch
from torch import nn
import tokenizer
#

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fullyconnected = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Compute queries, keys, and values
        queries = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scoring
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)

        # Pass through a final linear layer
        output = self.fullyconnected(attended_values)

        return output

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(EncoderLayer, self).__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        attn_output = self.self_attn(x)
        x = attn_output + x
        x = self.norm1(x)

        # Feed-forward
        ff_output = self.ff(x)
        x = ff_output + x
        x = self.norm2(x)

        return x

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head) for _ in range(n_layer)])

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, n_hidden, n_output):
        super().__init__()
        self.encoder = SimpleTransformerEncoder(vocab_size, n_embd, n_head, n_layer, block_size)
        self.decoder = TransformerDecoder(vocab_size, n_embd, n_head, n_layer, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.classifier = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x, mode='classifier'):
        if mode == 'classifier':
            x = self.encoder(x)
            x = x.mean(dim=1)
            x = self.classifier(x)
        elif mode == 'lm':
            x = self.decoder(x)
            x = self.lm_head(x)
        return x

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Compute queries, keys, and values
        queries = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Create a mask to prevent attending to future tokens
        mask = torch.tril(torch.ones(seq_length, seq_length)).view(1, 1, seq_length, seq_length).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)

        # Pass through a final linear layer
        output = self.fc(attended_values)

        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerDecoderLayer, self).__init__()
        self.masked_self_attn = MaskedSelfAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Masked self-attention
        attn_output = self.masked_self_attn(x)
        x = attn_output + x
        x = self.norm1(x)

        # Feed-forward
        ff_output = self.ff(x)
        x = ff_output + x
        x = self.norm2(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([TransformerDecoderLayer(n_embd, n_head, n_embd) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x