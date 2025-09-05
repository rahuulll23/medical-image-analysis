import os, argparse, torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.preprocessing import get_dataloaders, set_seed
from src.model import create_model
from src.utils import save_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/chest_xray")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()

def train(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, class_names = get_dataloaders(args.data_dir, args.batch_size)

    model = create_model(num_classes=len(class_names), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # training
        model.train()
        y_true, y_pred = [], []
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
        train_acc = accuracy_score(y_true, y_pred)

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.argmax(1).cpu().numpy())
        val_acc = accuracy_score(y_true, y_pred)
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), "saved_models/model.pth")
            print(f"✅ Best model saved (Val Acc={best_val_acc:.4f})")

if __name__ == "__main__":
    args = parse_args()
    train(args)
