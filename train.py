import torch
import torch.nn as nn

def train_model(model, train_loader, optimizer, criterion, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
    acc = 100 * correct / len(test_loader.dataset)
    print(f"Test Loss: {total_loss/len(test_loader):.4f}, Accuracy: {acc:.2f}%")
    return acc
