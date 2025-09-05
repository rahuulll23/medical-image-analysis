import torch, argparse
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import get_dataloaders
from src.model import create_model
from src.utils import plot_confusion_matrix

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/chest_xray")
    p.add_argument("--checkpoint", default="saved_models/model.pth")
    return p.parse_args()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_names = get_dataloaders(args.data_dir)
    model = create_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
