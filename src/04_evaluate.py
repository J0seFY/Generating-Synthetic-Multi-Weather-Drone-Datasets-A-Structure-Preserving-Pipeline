import os
import csv
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import clip
import piq
import pyiqa

#CONFIG =========
ORIGINAL_DIR = r"..\dataset"
GENERATED_DIR = r"..\output"
OUTPUT_CSV = r"..\evaluation_results.csv"

SCENARIOS = [
    ("night", "night time, darkness, cinematic lighting, street lights, 4k, realistic"),
    ("rain", "heavy rain, storm, wet surfaces, puddles, grey sky, rain streaks, 4k"),
    ("fog", "heavy fog, mist, low visibility, white atmosphere, dense haze"),
    ("sunny", "sunny day, clear blue sky, hard shadows, high contrast, noon"),
    ("sunset", "sunset, golden hour, orange sky, warm lighting, dramatic shadows")
]

EVAL_SIZE = (1024, 576)  # Tamaño fijo para evaluación

#DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#MODELS AND METRICS
loss_fn = lpips.LPIPS(net='alex').to(device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
piqe_metric = pyiqa.create_metric('piqe', device=device)
niqe_metric = pyiqa.create_metric('niqe', device=device)

#HELPERS
def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def to_tensor(img_np, for_lpips=False):
    t = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if for_lpips:
        t = t * 2 - 1
    return t.to(device)

#MAIN EVALUATION LOOP
rows = []

for scenario,scenario_prompt in SCENARIOS:
    scenario_path = os.path.join(GENERATED_DIR, scenario)
    if not os.path.exists(scenario_path):
        print(f"[WARN] Folder not found: {scenario_path}")
        continue
    
    print(f"Evaluating scenario: {scenario}")
    
    for gen_name in os.listdir(scenario_path):
        if not gen_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        # Extraer nombre base removiendo el sufijo del escenario 
        # Primero obtener extensión
        name_without_ext = os.path.splitext(gen_name)[0]
        ext = os.path.splitext(gen_name)[1]
        
        # Remover sufijo del escenario del final del nombre
        base_name = name_without_ext
        suffix = f"_{scenario}"
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]  # Remover sufijo del final
        
        base_name = base_name + ext  # Agregar extensión de vuelta
        
        original_path = os.path.join(ORIGINAL_DIR, base_name)
        generated_path = os.path.join(scenario_path, gen_name)

        if not os.path.exists(original_path):
            print(f"[SKIP] Original not found for {gen_name} (buscando: {base_name})")
            continue

        #Load images
        img_orig = load_image(original_path)
        img_gen = load_image(generated_path)
        
        # Redimensionar ambas imágenes a tamaño fijo para comparación consistente
        img_orig = np.array(Image.fromarray(img_orig).resize(EVAL_SIZE, Image.Resampling.LANCZOS))
        img_gen = np.array(Image.fromarray(img_gen).resize(EVAL_SIZE, Image.Resampling.LANCZOS))

        #  SSIM 
        ssim_score = ssim(img_orig, img_gen, channel_axis=2, data_range=255)

        #  PSNR 
        psnr_score = psnr(img_orig, img_gen, data_range=255)

        #  LPIPS 
        t_orig = to_tensor(img_orig, for_lpips=True)
        t_gen  = to_tensor(img_gen, for_lpips=True)

        #  CLIP & LPIPS & PIQE under no_grad to save VRAM 
        with torch.no_grad():
            lpips_score = loss_fn(t_orig, t_gen).item()

            #CLIP imagen a imagen
            img1 = clip_preprocess(Image.fromarray(img_orig)).unsqueeze(0).to(device)
            img2 = clip_preprocess(Image.fromarray(img_gen)).unsqueeze(0).to(device)
            f1 = clip_model.encode_image(img1)
            f2 = clip_model.encode_image(img2)
            clip_score = torch.cosine_similarity(f1, f2).item()

            #CLIP TEXT-TO-IMAGE
            text_tokens = clip.tokenize(scenario_prompt).to(device)
            f_text = clip_model.encode_text(text_tokens)            # (1,512)
            f_img  = clip_model.encode_image(img2)                  # imagen generada
            clip_text_image = torch.cosine_similarity(f_text, f_img).item()

            t_gen_nr = to_tensor(img_gen, for_lpips=False)
            brisque_score = piq.brisque(t_gen_nr, data_range=1.0).item()
            piqe_score = piqe_metric(t_gen_nr).item()
            niqe_score = niqe_metric(t_gen_nr).item()

        #  Save row 
        rows.append([
            base_name,
            scenario,
            round(ssim_score, 4),
            round(psnr_score, 2),
            round(lpips_score, 4),
            round(clip_score, 4),
            round(clip_text_image, 4),
            round(brisque_score, 2),
            round(piqe_score, 2),
            round(niqe_score, 2)
        ])

        print(f"OK -> {gen_name}")

# SAVE CSV 
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_name",
        "scenario",
        "SSIM",
        "PSNR_dB",
        "LPIPS",
        "CLIP_similarity",
        "CLIP_text2image",
        "BRISQUE",
        "PIQE",
        "NIQE"
    ])
    writer.writerows(rows)

print("DONE")
print("Results saved to:", OUTPUT_CSV)
