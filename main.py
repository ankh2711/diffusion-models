import json
import argparse
import os

import subprocess
from PIL import Image
import torch
from torchvision import transforms
from efficientnet.models.efficientnet import EfficientNet, params
from collections import OrderedDict
from torch import nn
import numpy as np
import cv2


def load_classifier(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = EfficientNet(1.0, 1.0, 0.2)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 7), nn.Softmax(dim=1))

    state_dict = checkpoint['model']
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model

def load_dataset_classifier(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = EfficientNet(1.0, 1.0, 0.2)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 3), nn.Softmax(dim=1))

    state_dict = checkpoint['model']
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model

def classify_image_no_mask(image, model):
    tfms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    img_t = tfms(image.convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        probs = model(img_t).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())

    labels_map = [
        '0_Periodic_noise',
        '1_Grid',
        '2_Stripes',
        '3_Gaussian_noise',
        '4_Half',
        '5_Rectangle',
        '6_Border'
    ]

    return labels_map[pred_idx], float(probs[pred_idx].item())

def generate_gaussian_periodic_mask(image):
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    mask = np.where(img_np == 0, 0, 255).astype(np.uint8)

    H, W = mask.shape
    img_area = H * W
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask==0).astype(np.uint8), 8)
    final_mask = np.ones((H, W), dtype=np.uint8) * 255

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 1:
            final_mask[labels == i] = 0

    return final_mask

def generate_stripes_mask(image):
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    mask = np.where(img_np == 0, 0, 255).astype(np.uint8)

    H, W = mask.shape
    img_area = H * W
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask==0).astype(np.uint8), 8)

    final_mask = np.ones((H, W), dtype=np.uint8) * 255

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 1:
            final_mask[labels == i] = 0

    return final_mask

def generate_half_border_rectangle_grid_mask(image, min_area_ratio=0.01):
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    mask = np.where(img_np == 0, 0, 255).astype(np.uint8)
    H, W = mask.shape
    img_area = H * W
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask == 0).astype(np.uint8), 8)

    final_mask = np.ones((H, W), dtype=np.uint8) * 255

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_ratio * img_area:
            final_mask[labels == i] = 0

    return final_mask

def generate_mask(image, label):
    if label in ['1_Grid', '4_Half', '5_Rectangle', '6_Border']:
        mask = generate_half_border_rectangle_grid_mask(image)

    elif label in ['0_Periodic_noise', '3_Gaussian_noise']:
        mask = generate_gaussian_periodic_mask(image)

    else:
        mask = generate_stripes_mask(image)

    return Image.fromarray(mask)

def classify_dataset(image, model):

    tfms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    img_t = tfms(image.convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        probs = model(img_t).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())

    labels_map = [
        '0_Celeba',
        '1_Imagenet',
        '2_Places2'
    ]

    return labels_map[pred_idx], float(probs[pred_idx].item())

def choose_model(pred_label, dataset_label, preference="quality"):

    # порядок по скорости
    FAST_PRIORITY = ["DiffPIR", "RePaint", "CoPaint"]

    # порядок по качеству
    if pred_label in ['0_Periodic_noise', '1_Grid', '2_Stripes', '4_Half', '6_Border']:
        quality_order = ["CoPaint", "RePaint", "DiffPIR"]

    elif pred_label == '3_Gaussian_noise':
        quality_order = ["RePaint", "DiffPIR", "CoPaint"]

    elif pred_label == '5_Rectangle':
        quality_order = ["DiffPIR", "RePaint", "CoPaint"]

    else:
        quality_order = ["RePaint", "DiffPIR", "CoPaint"]

    # допустимые модели
    if pred_label in ['0_Periodic_noise', '1_Grid', '2_Stripes', '6_Border']:
        suitable = ["CoPaint", "RePaint", "DiffPIR"]

    elif pred_label == '3_Gaussian_noise':
        suitable = ["RePaint", "DiffPIR"]

    else:
        suitable = ["CoPaint", "DiffPIR"]

    if dataset_label != "0_Celeba":
        suitable = [m for m in suitable if m != "CoPaint"]

    priority = FAST_PRIORITY if preference == "fast" else quality_order

    for m in priority:
        if m in suitable:
            return m

    return suitable[0] if suitable else "RePaint"

def choose_diffpir_model(dataset_label):
    if dataset_label == "Celeba":
        return "DiffPIR/model_zoo/diffusion_ffhq_10m.pt"
    else:
        return "DiffPIR/model_zoo/imagenet256.pt"

def auto_inpaint(image_path, classifier_path, dataset_classifier_path, repaint_config_path, preference):

    classifier_model = load_classifier(classifier_path)
    dataset_model = load_dataset_classifier(dataset_classifier_path)

    image = Image.open(image_path).convert('RGB')

    # 1. классификация искажения
    label, conf = classify_image_no_mask(image, classifier_model)
    print(f"Тип искажения: {label} ({conf:.2f})")

    # 2. маска
    mask = generate_mask(image, label)

    # сохранить маску
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs("generated_masks", exist_ok=True)

    mask_path = f"generated_masks/{base_name}_mask.png"
    mask.save(mask_path)

    # print(f"Маска сохранена: {mask_path}")

    # 3. классификация датасета
    dataset_label, dataset_conf = classify_dataset(image, dataset_model)
    print(f"Тип изображения: {dataset_label} ({dataset_conf:.2f})")

    # 4. выбор модели
    chosen_model = choose_model(label, dataset_label, preference)
    print(f"Выбрана модель ({preference}): {chosen_model}")

    diffpir_model = choose_diffpir_model(dataset_label)

    # --- 5. восстановление ---
    if chosen_model == "RePaint":
        subprocess.run([
            "python", "RePaint/test.py",
            "--conf_path", repaint_config_path,
            "--image_path", image_path,
            "--mask_path", mask_path
        ], check=True)
        restored = Image.open("result/inpainted/test_image.png")

    elif chosen_model == "DiffPIR":
        subprocess.run([
            "python", "DiffPIR/main_diffpir.py",
            "--input", image_path,
            "--mask", mask_path,
            "--model_path", diffpir_model
        ], check=True)
        restored = Image.open("result/diffpir/restored.png")

    elif chosen_model == "CoPaint":
        subprocess.run([
            "python", "CoPaint/main.py",
            "--input", image_path,
            "--mask", mask_path,
            "--model_path", "CoPaint/checkpoints/celeba256_250000.pt"
        ], check=True)
        restored = Image.open("result/copaint/copaint_restored.png")

    return restored

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, simpledialog

    root = tk.Tk()
    root.withdraw()

    img_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not img_path:
        print("Изображение не выбрано.")
        exit()

    preference = simpledialog.askstring(
        "Предпочтение",
        "1 — качество\n2 — скорость\nВведите 1 или 2:"
    )

    if preference == "2":
        preference = "fast"
    else:
        preference = "quality"

    classifier_path = "distortion_classifier.pth"
    dataset_classifier = "data_type_classifier.pth"
    config_path = "RePaint/confs/imagenet.yaml"

    restored_img = auto_inpaint(
        img_path,
        classifier_path,
        dataset_classifier,
        config_path,
        preference
    )

    restored_img.save("restored_result.png")
    print("Результат сохранен как restored_result.png")
