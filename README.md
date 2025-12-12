# Proyecto Tópicos Avanzados de IA

## Requisitos
- Python 3.11
- GPU NVIDIA (recomendado) con drivers/CUDA 12.6 si usas los wheels CUDA de PyTorch

## Instalación
```powershell
# Crear venv
python -m venv venv

# Activar venv (PowerShell)
& "D:\proyecto topicos avanzados de ia\venv\Scripts\Activate.ps1"

# Instalar dependencias
pip install -r requirements.txt
# Si quieres CUDA 12.6 para PyTorch:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Scripts y uso

### 01_filter_dataset.py
- Filtra imágenes redundantes mediante SSIM.
- Usa resize a 960x540 y compara en RGB.
- Configura en el script: `INPUT_DIR`, `OUTPUT_DIR`, `THRESHOLD`, `FIXED_SIZE`, `MAX_WORKERS`.
- Ejecuta:
```powershell
python src\01_filter_dataset.py
```

### 02_generate_dataset.py
- Genera variaciones climáticas con ControlNet (Canny) + Stable Diffusion v1.5.
- Optimizado para VRAM baja (cpu_offload, vae_slicing/tiling). Redimensiona entrada máx 1024 px.
- Configura en el script: `INPUT_DIR` (dataset filtrado), `OUTPUT_DIR`, `SEED`.
- Ejecuta:
```powershell
python src\02_generate_dataset.py
```

### 03_daam.py
- Genera imágenes con ControlNet y produce mapas de atención (DAAM).
- Incluye offload y slicing para reducir VRAM; ajusta rutas de entrada/salida en el script.
- Ejecuta:
```powershell
python src\03_daam.py
```

### 04_evaluate.py
- Evalúa imágenes generadas vs originales con SSIM, PSNR, LPIPS, CLIP (imagen y texto-imagen), BRISQUE, PIQE y NIQE.
- Redimensiona ambas imágenes a 1024x576 para comparación consistente.
- Configura en el script: `ORIGINAL_DIR`, `GENERATED_DIR`, `OUTPUT_CSV`.
- Ejecuta:
```powershell
python src\04_evaluate.py
```

## Notas
- Métricas sin referencia: BRISQUE (piq) y PIQE/NIQE (pyiqa).
- Ajusta rutas en los scripts según tu estructura.
- Para no llenar el disco C:, puedes mover la caché de Hugging Face:
```python
import os
os.environ['HF_HOME'] = r'D:\proyecto topicos avanzados de ia\.cache\huggingface'
os.environ['TRANSFORMERS_CACHE'] = r'D:\proyecto topicos avanzados de ia\.cache\huggingface\transformers'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'D:\proyecto topicos avanzados de ia\.cache\huggingface\hub'
```
