import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from daam import trace

# --- CONFIGURACIÓN ---
TEST_IMAGE_PATH = r"..\dataset\DJI_20250422114842_0003_frame_000000.jpg"
OUTPUT_DIR = r"..\xai_output"
PROMPT = "heavy rain, storm, wet surfaces, puddles, grey sky, rain streaks, 4k"
TOKENS_TO_VISUALIZE = ["heavy rain", "puddles", "storm", "grey sky", "rain streaks", "wet surfaces"]

def run_xai():
    
    # 1. Cargar ControlNet (Por defecto carga en float32)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny"
    )
    
    # 2. Cargar Pipeline (Por defecto carga en float32)
    # Sin torch_dtype, sin safety checker para ahorrar RAM
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet,
        safety_checker=None, 
        requires_safety_checker=False
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Mantiene el modelo en RAM y sube capas a la GPU solo cuando se usan.
    pipe.enable_model_cpu_offload() 
    
    # Procesa la atención por partes para no explotar la memoria
    pipe.enable_attention_slicing()
    
    # Procesa el VAE por trozos. 
    pipe.enable_vae_tiling() 
    
    #PROCESAMIENTO IMAGEN 
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"No se encuentra: {TEST_IMAGE_PATH}")
        
    original_image = Image.open(TEST_IMAGE_PATH).convert("RGB").resize((512, 512))
    image_cv = np.array(original_image)
    image_cv = cv2.Canny(image_cv, 100, 200)
    image_cv = np.stack([image_cv] * 3, axis=2)
    image_canny = Image.fromarray(image_cv)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generando con DAAM")
    
    with trace(pipe) as tc:
        # Generator en CPU para máxima compatibilidad
        output = pipe(
            PROMPT, 
            image=image_canny, 
            num_inference_steps=20,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]
        
        output.save(os.path.join(OUTPUT_DIR, "xai_generated_result.jpg"))
        print("magen guardada.")

        print("Generando mapas de calor...")
        # Nota: Usamos la API moderna de DAAM
        global_heat_map = tc.compute_global_heat_map(prompt=PROMPT)
        
        for token in TOKENS_TO_VISUALIZE:
            try:
                heat_map = global_heat_map.compute_word_heat_map(token)
                
                plt.figure(figsize=(10, 10))
                heat_map.plot_overlay(output)
                plt.axis('off')
                plt.title(f"Attention: '{token}'")
                
                save_path = os.path.join(OUTPUT_DIR, f"heatmap_{token}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                print(f"   -> Mapa guardado: {token}")
            except Exception as e:
                print(f"   [ERROR] Fallo visualizando '{token}': {e}")

if __name__ == "__main__":
    run_xai()