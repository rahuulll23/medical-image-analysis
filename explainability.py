import os, argparse, torch, numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.preprocessing import get_dataloaders
from src.model import create_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/chest_xray")
    p.add_argument("--checkpoint", default="saved_models/model.pth")
    p.add_argument("--out_dir", default="saved_models/cams")
    return p.parse_args()

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_names = get_dataloaders(args.data_dir, batch_size=1)
    model = create_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])


    for i, (img, label) in enumerate(test_loader):
        if i >= 10: break
        inp = img.to(device)
        grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(label.item())])[0, :]
        img_np = img.squeeze().permute(1,2,0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max()-img_np.min())
        cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        Image.fromarray(cam_img).save(os.path.join(args.out_dir, f"cam_{i}.png"))

if __name__ == "__main__":
    args = parse_args()
    run(args)
