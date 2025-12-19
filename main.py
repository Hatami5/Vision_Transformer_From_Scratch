import torch
import config
from dataset import get_dataloaders
from model.vit import VisionTransformer
from train import train_one_epoch
from eval import evaluate

torch.manual_seed(config.SEED)

train_loader, test_loader, classes = get_dataloaders()

model = VisionTransformer(config).to(config.DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

for epoch in range(config.EPOCHS):
    loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, config.DEVICE
    )
    test_acc = evaluate(model, test_loader, config.DEVICE)

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Train={train_acc:.4f}, Test={test_acc:.4f}")
