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
# from CoPaint.guided_diffusion.ddim import R_DDIMSampler

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

def classify_image(image, mask, model):
    img_np = np.array(image).astype(np.float32)
    mask_np = np.array(mask).astype(np.float32) / 255.0

    if mask_np.shape[:2] != img_np.shape[:2]:
        mask = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
            (img_np.shape[1], img_np.shape[0]), Image.NEAREST)
        mask_np = np.array(mask).astype(np.float32) / 255.0

    if mask_np.ndim == 2:
        mask_np = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
    elif mask_np.shape[2] == 4 and img_np.shape[2] == 3:
        mask_np = mask_np[:, :, :3]

    img_masked = img_np * mask_np
    img_masked = Image.fromarray(img_masked.astype(np.uint8)).convert('RGB')

    tfms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    img_t = tfms(img_masked).unsqueeze(0)
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

def classify_dataset(image, mask, model):
    img_np = np.array(image).astype(np.float32)
    mask_np = np.array(mask).astype(np.float32) / 255.0

    if mask_np.shape[:2] != img_np.shape[:2]:
        mask = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
            (img_np.shape[1], img_np.shape[0]), Image.NEAREST)
        mask_np = np.array(mask).astype(np.float32) / 255.0

    if mask_np.ndim == 2:
        mask_np = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
    elif mask_np.shape[2] == 4 and img_np.shape[2] == 3:
        mask_np = mask_np[:, :, :3]

    img_masked = img_np * mask_np
    img_masked = Image.fromarray(img_masked.astype(np.uint8)).convert('RGB')

    tfms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = tfms(img_masked).unsqueeze(0)
    with torch.no_grad():
        probs = model(img_t).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    labels_map = [
        'Celeba',
        'Places2',
        'Imagenet'
    ]
    return labels_map[pred_idx], float(probs[pred_idx].item())

def choose_model(pred_label):
    if pred_label in ['4_Half', '5_Rectangle', '6_Border']:
        return "RePaint"
    elif pred_label in ['0_Periodic_noise', '3_Gaussian_noise']:
        return "DiffPIR"
    else:
        return "CoPaint"

def choose_diffpir_model(dataset_label):
    if dataset_label == "Celeba":
        return "DiffPIR/model_zoo/diffusion_ffhq_10m.pt"
    else:
        return "DiffPIR/model_zoo/imagenet256.pt"

def auto_inpaint(image_path, mask_path, classifier_path, dataset_classifier_path, repaint_config_path):
    classifier_model = load_classifier(classifier_path)
    dataset_model = load_dataset_classifier(dataset_classifier_path)
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    label, conf = classify_image(image, mask, classifier_model)
    print(f"Тип искажения: {label} (уверенность: {conf:.2f})")
    chosen_model = choose_model(label)
    print(f"Выбрана модель: {chosen_model}")
    dataset_label, dataset_conf = classify_dataset(image, mask, dataset_model)
    print(f"Тип изображения: {dataset_label} ({dataset_conf:.2f})")
    diffpir_model = choose_diffpir_model(dataset_label)
    # print(f"DiffPIR модель: {diffpir_model}")

    if chosen_model == "RePaint":
        subprocess.run([
            "python", "RePaint/test.py",
            "--conf_path", repaint_config_path,
            "--image_path", image_path,
            "--mask_path", mask_path
        ], check=True)
        restored_path = "result/inpainted/test_image.png"
        restored = Image.open(restored_path)
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

# if __name__ == "__main__":
#     img_path = "input/image/test_image.png"
#     mask_path = "input/mask/test_mask.png"
#     classifier_path = "checkpoint2911.pth"
#     config_path = "RePaint/confs/imagenet.yaml"
#
#     restored_img = auto_inpaint(img_path, mask_path, classifier_path, config_path)
#     restored_img.save("restored_result.png")

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    # Создаем скрытое главное окно
    root = tk.Tk()
    root.withdraw()  # Скрыть главное окно

    # Открываем диалог выбора изображения
    img_path = filedialog.askopenfilename(
        title="Выберите изображение для восстановления",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not img_path:
        print("Изображение не выбрано. Завершение работы.")
        exit()

    # Открываем диалог выбора маски
    mask_path = filedialog.askopenfilename(
        title="Выберите маску для восстановления",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not mask_path:
        print("Маска не выбрана. Завершение работы.")
        exit()

    classifier_path = "checkpoint2911.pth"
    dataset_classifier = "classifier.pth"
    config_path = "RePaint/confs/imagenet.yaml"

    restored_img = auto_inpaint(img_path, mask_path, classifier_path, dataset_classifier, config_path)
    restored_img.save("restored_result.png")
    print("Результат сохранен как restored_result.png")
