import torch
import torch.nn as nn
from model import CNNWithAttention
from utils import get_data
from train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_loader, test_loader = get_data(batch_size=128)
    model = CNNWithAttention().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_model(model, train_loader, optimizer, criterion, device, epochs=20)
    evaluate_model(model, test_loader, criterion, device)
    MODEL_PATH = "runs/cifar10_cnn_attention.pth"
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    main()
