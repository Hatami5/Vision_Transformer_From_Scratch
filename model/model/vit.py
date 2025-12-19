import torch.nn as nn
from .patch_embedding import PatchEmbedding
from .transformer import TransformerEncoder

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            config.IMAGE_SIZE,
            config.PATCH_SIZE,
            config.CHANNELS,
            config.EMBED_DIM
        )

        self.encoder = nn.Sequential(
            *[TransformerEncoder(
                config.EMBED_DIM,
                config.NUM_HEADS,
                config.MLP_DIM,
                config.DROP_RATE
            ) for _ in range(config.DEPTH)]
        )

        self.norm = nn.LayerNorm(config.EMBED_DIM)
        self.head = nn.Linear(config.EMBED_DIM, config.NUM_CLASSES)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0])
