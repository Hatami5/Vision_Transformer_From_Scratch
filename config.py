#config.py
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 10
LR = 3e-4

IMAGE_SIZE = 32
PATCH_SIZE = 4
CHANNELS = 3
NUM_CLASSES = 10

EMBED_DIM = 256
DEPTH = 6
NUM_HEADS = 8
MLP_DIM = 512
DROP_RATE = 0.1

SEED = 42
