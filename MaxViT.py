import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class MaxViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MaxViTBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.transformer = TransformerBlock(embed_size=out_channels, heads=4, forward_expansion=4, dropout=0.1)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        B, C, H, W = x.shape

        # Görüntüyü transformer block için yeniden şekillendir
        x = x.flatten(2).permute(0, 2, 1)  # (B, num_patches, embed_size)
        
        x = self.transformer(x, x, x, mask=None)

        # Transformer'dan dönen veriyi yeniden şekillendir
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.conv2(x)
        return x

class MaxViT(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(MaxViT, self).__init__()
        self.layer1 = MaxViTBlock(in_channels, 64, stride=2)
        self.layer2 = MaxViTBlock(64, 128, stride=2)
        self.layer3 = MaxViTBlock(128, 256, stride=2)
        self.layer4 = MaxViTBlock(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Örnek kullanım
model = MaxViT(in_channels=3, num_classes=10)
x = torch.randn(1, 3, 224, 224)  # Rastgele bir giriş görüntüsü
output = model(x)
print(output.shape)  # Çıkış boyutunu kontrol et

'''
ConvBlock Sınıfı: Konvolüsyonel katman, batch normalization ve ReLU aktivasyon fonksiyonunu içerir.
SelfAttention Sınıfı: Transformer'ın dikkat mekanizması, çoklu başlarla birlikte.
TransformerBlock Sınıfı: Transformer bloğu, self-attention ve feed-forward katmanlarını içerir.
MaxViTBlock Sınıfı: MaxViT bloğu, bir konvolüsyonel blok, transformer bloğu ve ardından başka bir konvolüsyonel blok içerir.
MaxViT Sınıfı: MaxViT modelinin genel yapısı, çeşitli MaxViT bloklarından oluşur ve sonunda sınıflandırma için bir tam bağlı katman bulunur.

'''





