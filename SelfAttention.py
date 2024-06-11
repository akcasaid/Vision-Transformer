import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Değerleri, anahtarları ve sorguları farklı kafalara bölmek
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Matris çarpımları ile sorgu ve anahtarların çarpımı
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Maske uygulanıyorsa negatif sonsuz eklemek
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Softmax ile normalizasyon
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Değerler ile ağırlıklı toplam
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # Son katman: Dense layer
        out = self.fc_out(out)
        return out

# Örnek kullanım
embed_size = 256
heads = 8
self_attention = SelfAttention(embed_size, heads)

# Rastgele girişler
values = torch.randn(3, 10, embed_size)
keys = torch.randn(3, 10, embed_size)
queries = torch.randn(3, 10, embed_size)
mask = None

output = self_attention(values, keys, queries, mask)
print(output.shape)
