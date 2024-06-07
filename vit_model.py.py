import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    # Görüntü boyutları, yama boyutu, gömme boyutu ve diğer parametreler tanımlanıyor
    img_size = 224  # Görüntü boyutu
    patch_size = 16  # Her bir yamanın boyutu
    num_patches = (img_size // patch_size) ** 2  # Toplam yama sayısı
    num_classes = 1000  # Sınıf sayısı
    embed_dim = 768  # Gömme boyutu
    num_heads = 12  # Dikkat başlığı sayısı
    num_layers = 12  # Transformer katman sayısı
    hidden_dim = 3072  # Tam bağlı katmanın boyutu
    dropout_rate = 0.1  # Dropout oranı

class PatchEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # 3 kanallı girdi görüntüsünden gömme boyutuna yamalar çıkartan bir konvolüsyon katmanı
        self.proj = nn.Conv2d(3, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x):
        x = self.proj(x)  # Görüntüyü yamalara ayırır ve her yamayı gömme boyutuna dönüştürür (B, C, H, W)
        x = x.flatten(2)  # Yamaları düzleştirir (B, C, N)
        x = x.transpose(1, 2)  # Yamaların boyutlarını değiştirir (B, N, C)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # Sorgu, anahtar ve değer vektörlerini oluşturmak için tam bağlı katmanlar
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.num_heads = config.num_heads  # Başlık sayısı
        self.scale = config.embed_dim ** -0.5  # Ölçekleme faktörü

    def forward(self, x):
        B, N, C = x.shape  # Girdi boyutlarını alır
        # Sorgu, anahtar ve değerleri başlıklara ayırır ve boyutlarını değiştirir
        q = self.query(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        # Dikkat skorlarını hesaplar ve normalleştirir
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        # Son çıktıyı hesaplar
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return out



class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: Config):
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
        # Çoklu başlık dikkat bloğu
        x2 = self.norm1(x)
        x = x + self.attn(x2)
        # Feed-forward ağ
        x2 = self.norm2(x)
        x = x + self.ffn(x2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.patch_embedding = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + config.num_patches, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout_rate)
        self.transformer_blocks = nn.Sequential(*[TransformerEncoderBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


def create_model_vit():
    # Yapılandırma sınıfından bir nesne oluştur
    config = Config()

    # VisionTransformer modelini bu yapılandırma ile başlat
    model = VisionTransformer(config)

    # Modeli döndür
    return model


vit_model = create_model_vit()