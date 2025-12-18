from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE

def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_train
    )
    test_ds = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_ds.classes
