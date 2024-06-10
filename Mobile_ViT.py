import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileViTConfig:
    # Görüntü boyutları, yama boyutu, gömme boyutu ve diğer parametreler tanımlanıyor
    img_size = 224  # Görüntü boyutu
    patch_size = 16  # Her bir yamanın boyutu
    num_patches = (img_size // patch_size) ** 2  # Toplam yama sayısı
    num_classes = 1000  # Sınıf sayısı
    embed_dim = 768  # Gömme boyutu
    num_heads = 8  # Dikkat başlığı sayısı
    num_layers = 3  # Transformer katman sayısı
    hidden_dim = 512  # Tam bağlı katmanın boyutu
    dropout_rate = 0.1  # Dropout oranı

class PatchEmbedding(nn.Module):
    def __init__(self, config: MobileViTConfig):
        super().__init__()
        self.proj = nn.Conv2d(3, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x):
        x = self.proj(x)  # Görüntüyü yamalara ayırır ve her yamayı gömme boyutuna dönüştürür (B, C, H, W)
        x = x.flatten(2)  # Yamaları düzleştirir (B, C, N)
        x = x.transpose(1, 2)  # Yamaların boyutlarını değiştirir (B, N, C)
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, config: MobileViTConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(config.embed_dim, config.hidden_dim, kernel_size=3, padding=1)
        self.transformer = TransformerEncoderBlock(config)
        self.conv2 = nn.Conv2d(config.hidden_dim, config.embed_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.conv2(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: MobileViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout_rate)
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2)
        x2 = self.norm2(x)
        x = x + self.ffn(x2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config: MobileViTConfig):
        super().__init__()
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.num_heads = config.num_heads
        self.scale = config.embed_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return out

class MobileViT(nn.Module):
    def __init__(self, config: MobileViTConfig):
        super().__init__()
        self.patch_embedding = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + config.num_patches, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout_rate)
        self.mobilevit_blocks = nn.Sequential(*[MobileViTBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.mobilevit_blocks(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

def create_model_mobilevit():
    config = MobileViTConfig()
    model = MobileViT(config)
    return model

mobilevit_model = create_model_mobilevit()
