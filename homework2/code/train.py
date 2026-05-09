import json
import os

import torch
import torch.nn as nn

from data import make_dataloaders
from model import Fashion_Mnist_Model

def evaluate(model: Fashion_Mnist_Model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc

def train(
    art_dir:str = "tmp/fashion_mnist",
    num_class:int = 10,
    dropout:float = 0.3,
    num_epoch:int = 100,
    batch_size:int = 256,
    lr:float = 1e-3,
)->None:
    os.makedirs(art_dir,exist_ok=True)

    config = {
        "num_class":num_class,
        "dropout":dropout,
        "num_epoch":num_epoch,
        "batch_size":batch_size,
        "lr":lr,
    }
    with open(os.path.join(art_dir,"config.json"),"w") as f:
        json.dump(config,f,indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train,test = make_dataloaders()

    model = Fashion_Mnist_Model(num_class,dropout).to(device)
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    loss_f = nn.CrossEntropyLoss()

    for epoch in range(1,num_epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for image,label in  train:
            image,label = image.to(device),label.to(device)
            optim.zero_grad()
            output = model.forward(image)
            loss = loss_f(output,label)
            loss.backward()
            optim.step()

            total_loss += loss.item() * label.size(0)
            correct += (output.argmax(1)==label).sum().item()
            total += label.size(0)

        avg_loss = total_loss / total
        train_acc = correct / total * 100
        print(f"Epoch [{epoch}/{num_epoch}]  loss: {avg_loss:.4f}  train acc: {train_acc:.2f}%")

    print()

    evaluate(model, test, device)

    torch.save(model.state_dict(), os.path.join(art_dir, "model.pt"))
    print(f"Model saved to {art_dir}/model.pt")


        
            
