import torchvision
import json
import os

import torch
import torch.nn as nn

from data import FASHION_MNIST_TRANSFORM
from train import Fashion_Mnist_Model

def _load_model(art_dir: str, device: torch.device) -> Fashion_Mnist_Model:
    with open(os.path.join(art_dir, "config.json")) as f:
        config = json.load(f)
    model = Fashion_Mnist_Model(
        num_class=config["num_class"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(art_dir, "model.pt"), map_location=device))
    model.eval()
    return model

def infer(art_dir:str = "tmp/data",indent:int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model("tmp/fashion_mnist",device)

    ds = torchvision.datasets.FashionMNIST(root="tmp/data",train=False,download=True,transform=FASHION_MNIST_TRANSFORM)
    image,label = ds[indent]

    with torch.no_grad():
        output = model(image.unsqueeze(1).to(device))
        predict = output.argmax(1).item()

    print(f"pre{predict}|exp{label}")

def evaluate(art_dir: str = "/tmp/mnist_cnn"):
    from .data import make_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(art_dir, "config.json")) as f:
        config = json.load(f)

    model = _load_model(art_dir, device)
    _, test_loader = make_dataloaders(config["batch_size"])

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct / total * 100:.2f}% ({correct}/{total})")
    
    
