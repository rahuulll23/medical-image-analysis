import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from src.model import create_model
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.title("Chest X-Ray Pneumonia Detector")

MODEL_PATH = "saved_models/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():

    model = create_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

uploaded_file = st.file_uploader("Upload Chest X-ray (png/jpg)", type=["png","jpg","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded", width=350)

    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inp)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    st.write("Prediction:", "PNEUMONIA" if probs[1] > probs[0] else "NORMAL")
    st.write(f"Probabilities -> Normal: {probs[0]:.3f}, Pneumonia: {probs[1]:.3f}")

    # Grad-CAM
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(1)])[0, :]
    img_np = np.array(img.resize((224,224))).astype(np.float32)/255.0
    cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    st.image(cam_img, caption="Grad-CAM Overlay", width=350)
