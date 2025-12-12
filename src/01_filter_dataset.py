import os
import cv2
import shutil
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# CONFIGURACION
# Usamos r"" para evitar problemas con los backslashes en Windows
INPUT_DIR = r"..\dataset"
OUTPUT_DIR = r"..\filtered_dataset"
THRESHOLD = 0.6
FIXED_SIZE = (960, 540)  # Tamaño fijo para el cálculo de SSIM
MAX_WORKERS = 4  # Número de threads paralelos

def process_image(args):
    """
    Procesa una imagen: Carga BGR -> Convierte a Grayscale -> Redimensiona.
    Retorna (img_name, img_gray_resized, img_path)
    """
    img_name, img_path = args
    try:
        # OpenCV carga en BGR por defecto
        current_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if current_img is None:
            return None
        
        # Convertir a escala de grises
        current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar la imagen en escala de grises
        current_resized = cv2.resize(current_gray, FIXED_SIZE)
        
        return (img_name, current_resized, img_path)
    except Exception as e:
        print(f"Error procesando {img_name}: {e}")
        return None

def filter_redundant_images():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: No existe el directorio de entrada: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Filtrar solo extensiones de imagen válidas
    images = sorted([
        f for f in os.listdir(INPUT_DIR) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    if not images:
        print("No se encontraron imágenes en el directorio.")
        return

    print(f"Procesando {len(images)} imágenes en paralelo ({MAX_WORKERS} threads)...")
    
    # 1. Cargar y preprocesar en paralelo
    image_pairs = [(img_name, os.path.join(INPUT_DIR, img_name)) for img_name in images]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_image, image_pairs),
            total=len(image_pairs),
            desc="Cargando y convirtiendo a Grayscale"
        ))
    
    # Filtrar errores de carga
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No se pudieron cargar imágenes válidas.")
        return

    # 2. Comparación secuencial (SSIM requiere orden temporal)
    print(f"Comparando {len(valid_results)} imágenes válidas...")
    
    selected_count = 0
    last_image_data = None
    
    for img_name, current_resized, img_path in tqdm(valid_results, desc="Calculando SSIM y Filtrando"):
        is_distinct = False
        
        if last_image_data is None:
            # Siempre guardamos el primer frame
            is_distinct = True
        else:
            # SSIM en arrays 2D (Grayscale). No se necesita channel_axis.
            score = ssim(last_image_data, current_resized, data_range=255)
            
            # Si la similitud es menor al umbral, la imagen es distinta (útil)
            if score < THRESHOLD:
                is_distinct = True
        
        if is_distinct:
            shutil.copy2(img_path, os.path.join(OUTPUT_DIR, img_name))
            last_image_data = current_resized
            selected_count += 1

    print(f"FIN")
    print(f"Total imágenes procesadas: {len(images)}")
    print(f"Imágenes seleccionadas: {selected_count}")
    print(f"Ratio de retención: {(selected_count/len(images))*100:.2f}%")

if __name__ == "__main__":
    filter_redundant_images()