from torch.utils.data import DataLoader
from torchvision import datasets,transforms

FASHION_MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor()]
)

def make_dataloaders(
    batch_size: int = 256,
    data_root: str = "tmp/data"
)-> tuple[DataLoader,DataLoader]:
    trains_ds = datasets.FashionMNIST(root=data_root,transform=FASHION_MNIST_TRANSFORM,download=True,train=True)
    test_ds = datasets.FashionMNIST(root=data_root,transform=FASHION_MNIST_TRANSFORM,download=True,train=False)

    train_loader = DataLoader(trains_ds,shuffle=True,batch_size=batch_size)
    test_loader = DataLoader(test_ds,shuffle=False,batch_size=batch_size)

    return train_loader,test_loader
