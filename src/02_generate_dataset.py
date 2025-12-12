import torch
import cv2
import os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector

#configuracion
INPUT_DIR = r"..\filtered_dataset"
OUTPUT_DIR = r"..\output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42 

print(f"Using device: {DEVICE}")

# Definir los escenarios (Prompts)
SCENARIOS = [
    ("night", "night time, darkness, cinematic lighting, street lights, 4k, realistic"),
    ("rain", "heavy rain, storm, wet surfaces, puddles, grey sky, rain streaks, 4k"),
    ("fog", "heavy fog, mist, low visibility, white atmosphere, dense haze"),
    ("sunny", "sunny day, clear blue sky, hard shadows, high contrast, noon"),
    ("sunset", "sunset, golden hour, orange sky, warm lighting, dramatic shadows")
]

NEGATIVE_PROMPT = "cartoon, painting, illustration, blurry, distorted, low quality, bad anatomy"

# Cargar modelos
print("cargando Modelos en Float32")

# 1. Cargar ControlNet (Por defecto carga en float32)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny"
)

# 2. Cargar Pipeline (Por defecto carga en float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet,
    safety_checker=None,          # Desactiva censura
    requires_safety_checker=False 
)

# --- OPTIMIZACIONES PARA QUE QUEPA EN 4GB VRAM (AUNQUE SEA FLOAT32) ---
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Esto es obligatorio. Mantiene el modelo pesado en la RAM de tu PC
# y solo pasa trocitos a la GPU para procesar.
pipe.enable_model_cpu_offload() 

# Optimizaciones extra para el VAE
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

print("Modelo cargado en Float32.")
os.makedirs(OUTPUT_DIR, exist_ok=True)


images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Encontradas {len(images)} imágenes.")

for img_name in images:
    try:
        img_path = os.path.join(INPUT_DIR, img_name)
        original_image = Image.open(img_path).convert("RGB")
        
        # Redimensionar para mantener el consumo de vram bajo
        # Stable Diffusion prefiere múltiplos de 8. Reducimos a max 1024px.
        w, h = original_image.size
        if w > 1024 or h > 1024:
            ratio = min(1024/w, 1024/h)
            new_size = (int(w*ratio), int(h*ratio))
            original_image = original_image.resize(new_size, Image.LANCZOS)
        
        #Extraer bordes
        image_cv = np.array(original_image)
        image_cv = cv2.Canny(image_cv, 100, 200)
        image_canny = Image.fromarray(image_cv)
        
        base_name = os.path.splitext(img_name)[0]
        
        print(f"Procesando: {img_name}...")
        
        for scenario_name, prompt in SCENARIOS:
            save_path = os.path.join(OUTPUT_DIR, scenario_name)
            os.makedirs(save_path, exist_ok=True)
            
            #Generator en CPU para estabilidad en low-vram
            generator = torch.Generator("cpu").manual_seed(SEED)
            
            output = pipe(
                prompt,
                image=image_canny,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=20,         
                controlnet_conditioning_scale=1.0, 
                generator=generator
            ).images[0]
            
            save_file = f"{save_path}\\{base_name}_{scenario_name}.jpg"
            output.save(save_file)
            print(f"Guardado: {scenario_name}")

    except Exception as e:
        print(f"Error en {img_name}: {e}")
        torch.cuda.empty_cache() # Limpiar memoria si falla

print("Generación completa")