"""
=============================================================================
SISTEMA DE SEMÁFORO INTELIGENTE — ENTRENAMIENTO YOLOv8
=============================================================================
Detecta: person, bicycle, motorbike (+ alias motorcycle), car, bus, truck

FLUJO COMPLETO:
    1. Ejecutar primero:  python convertir_xml_a_yolo.py
    2. Luego entrenar:    python train_yolo.py

Requisitos:
    pip install ultralytics pyyaml

Hardware objetivo: NVIDIA RTX 3060 12 GB
=============================================================================
"""

# =============================================================================
# SECCIÓN 1: IMPORTACIONES
# ultralytics es la librería oficial de YOLOv8. Incluye modelo, entrenamiento,
# validación e inferencia en una sola API de alto nivel.
# =============================================================================
import os
import yaml
import torch
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


# =============================================================================
# SECCIÓN 2: CONFIGURACIÓN GENERAL
# Todos los parámetros del proyecto en un solo lugar.
# =============================================================================

# --- Archivo de configuración del dataset (generado por convertir_xml_a_yolo.py) ---
DATA_YAML = "data.yaml"

# --- Modelo base de YOLOv8 ---
# Opciones de menor a mayor capacidad (y tiempo de entrenamiento):
#   yolov8n.pt  → nano    (~3M params)  — más rápido, menos preciso
#   yolov8s.pt  → small   (~11M params) — buen balance para semáforos
#   yolov8m.pt  → medium  (~26M params) — recomendado con RTX 3060 12GB
#   yolov8l.pt  → large   (~44M params) — posible con batch bajo
#   yolov8x.pt  → xlarge  (~68M params) — necesita más VRAM
# Con 12 GB de VRAM, yolov8m es el punto óptimo calidad/velocidad.
MODEL_BASE  = "yolov8m.pt"

# --- Directorio donde YOLOv8 guarda resultados (pesos, métricas, gráficas) ---
PROJECT_DIR = "runs_semaforo"
RUN_NAME    = "yolov8m_semaforo_v1"

# --- Hiperparámetros de entrenamiento ---
NUM_EPOCHS  = 70           # YOLOv8 converge bien en 50-100 épocas
# RTX 3060 12 GB: batch 16 con yolov8m es cómodo.
# Si aparece OOM, bajar a 8. Subir a 32 con yolov8s/n.
BATCH_SIZE  = 16
# Tamaño de imagen: YOLO redimensiona internamente. 640 es el estándar.
# 1280 da más precisión en objetos pequeños pero duplica uso de VRAM.
IMG_SIZE    = 640
# Hilos de carga de datos. 8 es óptimo para CPUs modernas con RTX 3060.
# En Windows bajar a 0 o 4.
NUM_WORKERS = 8

# --- Hiperparámetros avanzados (con valores óptimos para tráfico urbano) ---
LR0         = 0.01         # Tasa de aprendizaje inicial
LRF         = 0.01         # Factor final de LR (lr_final = lr0 * lrf)
MOMENTUM    = 0.937        # Momentum SGD (estándar YOLO)
WEIGHT_DECAY= 0.0005       # Regularización L2
WARMUP_EPOCHS = 3          # Épocas de warmup antes de decay normal
# Augmentación — valores ajustados para escenas de tráfico:
MOSAIC      = 1.0          # Mosaico 4-en-1 (muy útil para objetos pequeños)
MIXUP       = 0.1          # MixUp entre imágenes (0=desactivado)
DEGREES     = 5.0          # Rotación máxima (°) — poca rotación en tráfico
TRANSLATE   = 0.1          # Traslación aleatoria
SCALE       = 0.5          # Escala aleatoria
FLIPLR      = 0.5          # Flip horizontal
FLIPUD      = 0.0          # Flip vertical (no tiene sentido en tráfico)
HSV_H       = 0.015        # Variación de matiz
HSV_S       = 0.7          # Variación de saturación
HSV_V       = 0.4          # Variación de brillo

# --- Umbral de confianza para inferencia ---
CONF_THRESH = 0.5
IOU_THRESH  = 0.45         # NMS IoU threshold

# --- Clases del proyecto ---
CLASSES = ["person", "bicycle", "motorbike", "car", "bus", "truck"]

# --- Dispositivo ---
DEVICE = 0 if torch.cuda.is_available() else "cpu"   # 0 = primera GPU
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] VRAM: {vram:.1f} GB")
else:
    print("[INFO] GPU no disponible, usando CPU")


# =============================================================================
# SECCIÓN 3: VERIFICACIÓN DEL ENTORNO
# Antes de entrenar se comprueba que todos los archivos necesarios existen
# y que el data.yaml tiene el formato correcto.
# =============================================================================

def verificar_entorno():
    """
    Verifica que el data.yaml existe y que las rutas de imágenes son válidas.
    Lanza un error descriptivo si algo falta, antes de comenzar el entrenamiento.
    """
    # Verificar data.yaml
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"No se encontró '{DATA_YAML}'.\n"
            "Ejecuta primero: python convertir_xml_a_yolo.py"
        )

    # Leer y validar el YAML
    with open(DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    required_keys = ["path", "train", "val", "nc", "names"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"El data.yaml no tiene la clave requerida: '{key}'")

    # Verificar que las carpetas de imágenes existen
    train_path = os.path.join(data["path"], data["train"])
    val_path   = os.path.join(data["path"], data["val"])

    for path, name in [(train_path, "train"), (val_path, "val")]:
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Carpeta de {name} no encontrada: {path}")
        n_imgs = len([f for f in os.listdir(path)
                      if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        print(f"  [✓] {name:5}: {n_imgs} imágenes en {path}")

    # Verificar consistencia de clases
    if data["nc"] != len(CLASSES):
        raise ValueError(
            f"Inconsistencia: data.yaml tiene nc={data['nc']} "
            f"pero CLASSES tiene {len(CLASSES)} entradas"
        )

    print(f"  [✓] data.yaml válido — {data['nc']} clases: {data['names']}")
    return True


# =============================================================================
# SECCIÓN 4: ENTRENAMIENTO
# YOLOv8 encapsula todo el ciclo de entrenamiento en model.train().
# Se pasan todos los hiperparámetros como argumentos con nombre.
# =============================================================================

def entrenar():
    """
    Carga el modelo base preentrenado (COCO) y lo afina (fine-tune)
    sobre el dataset del semáforo inteligente.

    YOLOv8 gestiona automáticamente:
      - Mixed Precision (AMP) en GPUs compatibles
      - Guardado de best.pt y last.pt
      - Gráficas de métricas (loss, mAP, precision, recall)
      - Exportación de resultados a CSV
    """
    print("\n" + "=" * 60)
    print("  INICIANDO ENTRENAMIENTO YOLOv8")
    print(f"  Modelo base : {MODEL_BASE}")
    print(f"  Épocas      : {NUM_EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Img size    : {IMG_SIZE}px")
    print(f"  Dispositivo : {'GPU ' + torch.cuda.get_device_name(0) if isinstance(DEVICE, int) else 'CPU'}")
    print("=" * 60)

    # ── Cargar modelo preentrenado ─────────────────────────────────────────
    # Si MODEL_BASE es "yolov8m.pt" y no existe localmente, se descarga
    # automáticamente desde los servidores de Ultralytics.
    # Los pesos están preentrenados en COCO (80 clases) — transfer learning.
    model = YOLO(MODEL_BASE)

    # ── Lanzar entrenamiento ───────────────────────────────────────────────
    results = model.train(
        # Dataset
        data        = DATA_YAML,
        # Entrenamiento
        epochs      = NUM_EPOCHS,
        patience    = 40,
        batch       = BATCH_SIZE,
        imgsz       = IMG_SIZE,
        workers     = NUM_WORKERS,
        device      = DEVICE,
        # Salida
        project     = PROJECT_DIR,
        name        = RUN_NAME,
        exist_ok    = False,       # False = crea nueva carpeta si ya existe
        save        = True,        # Guardar best.pt y last.pt
        save_period = 10,          # Guardar checkpoint extra cada N épocas
        # Optimizador
        optimizer   = "SGD",
        lr0         = LR0,
        lrf         = LRF,
        momentum    = MOMENTUM,
        weight_decay= WEIGHT_DECAY,
        warmup_epochs = WARMUP_EPOCHS,
        # Augmentación
        mosaic      = MOSAIC,
        mixup       = MIXUP,
        degrees     = DEGREES,
        translate   = TRANSLATE,
        scale       = SCALE,
        fliplr      = FLIPLR,
        flipud      = FLIPUD,
        hsv_h       = HSV_H,
        hsv_s       = HSV_S,
        hsv_v       = HSV_V,
        # Otros
        amp         = True,        # AMP automático — YOLOv8 lo activa si hay GPU compatible
        plots       = True,        # Generar gráficas de métricas y ejemplos
        verbose     = True,
    )

    return model, results


# =============================================================================
# SECCIÓN 5: VALIDACIÓN FINAL
# Tras el entrenamiento se evalúa el modelo con best.pt sobre el conjunto
# de validación y se imprimen métricas detalladas: mAP50, mAP50-95,
# precision y recall por clase.
# =============================================================================

def validar(model):
    """
    Evalúa el modelo entrenado sobre el conjunto de validación.

    Métricas principales que reporta YOLOv8:
      - Precision  : De todas las detecciones, cuántas son correctas
      - Recall     : De todos los objetos reales, cuántos fueron detectados
      - mAP@0.50   : Mean Average Precision con IoU ≥ 0.50
      - mAP@0.5:0.95: mAP promediado en IoU de 0.50 a 0.95 (métrica COCO)
    """
    print("\n[Validación] Evaluando best.pt sobre conjunto de validación...")

    metrics = model.val(
        data    = DATA_YAML,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        device  = DEVICE,
        verbose = True,
    )

    print("\n[Métricas finales]")
    print(f"  mAP@50     : {metrics.box.map50:.4f}")
    print(f"  mAP@50:95  : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")

    return metrics


# =============================================================================
# SECCIÓN 6: VISUALIZACIÓN DE PREDICCIONES
# Toma imágenes aleatorias de validación, corre inferencia y dibuja los
# bounding boxes predichos con etiqueta y confianza.
# =============================================================================

def visualizar_predicciones(model, num_samples: int = 6):
    """
    Genera una grilla de imágenes de validación con predicciones superpuestas.
    Guarda la imagen en el directorio de resultados del run.
    """
    # Leer rutas de imágenes de validación desde el YAML
    with open(DATA_YAML, "r") as f:
        data = yaml.safe_load(f)
    val_img_dir = os.path.join(data["path"], data["val"])

    img_files = [
        str(p) for p in Path(val_img_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not img_files:
        print("[Advertencia] No se encontraron imágenes de validación")
        return

    samples = random.sample(img_files, min(num_samples, len(img_files)))

    # Paleta de colores por clase
    palette = {
        "person":    "#FF4444",
        "bicycle":   "#44AAFF",
        "motorbike": "#FF8800",
        "car":       "#44FF44",
        "bus":       "#FF44FF",
        "truck":     "#FFFF44",
    }

    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for ax, img_path in zip(axes, samples):
        # Inferencia — YOLOv8 devuelve objetos Results
        results = model.predict(
            source  = img_path,
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            device  = DEVICE,
            verbose = False,
        )[0]

        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(Path(img_path).name, fontsize=7)

        # Dibujar cada detección
        for box in results.boxes:
            cls_id = int(box.cls.item())
            score  = box.conf.item()
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
            color  = palette.get(cls_name, "#FFFFFF")

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 4,
                f"{cls_name} {score:.2f}",
                color=color, fontsize=8, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1)
            )

    # Ocultar ejes sobrantes
    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.suptitle("YOLOv8 — Predicciones sobre Validación", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(PROJECT_DIR, RUN_NAME, "sample_predictions.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualización] Guardada en: {out_path}")


# =============================================================================
# SECCIÓN 7: EXPORTACIÓN DEL MODELO
# YOLOv8 puede exportar a múltiples formatos para despliegue.
# Para el semáforo inteligente en tiempo real, ONNX o TensorRT son ideales.
# =============================================================================

def exportar_modelo(model, formato: str = "onnx"):
    """
    Exporta el modelo entrenado a un formato de despliegue.

    Formatos útiles para el semáforo inteligente:
      - "onnx"      : Compatible con OpenCV DNN, ONNX Runtime, multiplataforma
      - "torchscript": PyTorch nativo, sin dependencias externas
      - "engine"    : TensorRT — máxima velocidad en GPUs NVIDIA (requiere TensorRT)
      - "openvino"  : Intel OpenVINO para CPUs Intel
      - "tflite"    : TensorFlow Lite para dispositivos embebidos

    Args:
        model   : Modelo YOLOv8 entrenado
        formato : Formato de exportación (default: "onnx")
    """
    print(f"\n[Exportación] Exportando modelo a formato: {formato.upper()}")
    export_path = model.export(
        format  = formato,
        imgsz   = IMG_SIZE,
        dynamic = False,     # False = tamaño fijo, más rápido en producción
        simplify = True,     # Simplificar grafo ONNX (solo aplica para onnx)
    )
    print(f"  [✓] Modelo exportado en: {export_path}")
    return export_path


# =============================================================================
# SECCIÓN 8: PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # Semilla de reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ── Paso 1: Verificar entorno ──────────────────────────────────────────
    print("\n[Paso 1] Verificando entorno...")
    verificar_entorno()

    # ── Paso 2: Entrenar ───────────────────────────────────────────────────
    print("\n[Paso 2] Entrenando YOLOv8m...")
    model = entrenar()

    # ── Paso 3: Validar con best.pt ────────────────────────────────────────
    print("\n[Paso 3] Cargando best.pt para validación final...")

    if hasattr(model, 'trainer') and model.trainer is not None:
        best_weights = model.trainer.best
        print(f"Usando pesos encontrados en: {best_weights}")
    else:
        # Fallback manual si no viene del entrenamiento directo
        best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")

    # ESTO DEBE ESTAR FUERA DEL ELSE (alineado con el 'if')
    model_best = YOLO(best_weights)
    metrics    = validar(model_best)

    # ── Paso 4: Visualizar predicciones ────────────────────────────────────
    print("\n[Paso 4] Generando visualizaciones...")
    visualizar_predicciones(model_best, num_samples=6)

    # ── Paso 5: Exportar a ONNX para despliegue ────────────────────────────
    print("\n[Paso 5] Exportando a ONNX...")
    exportar_modelo(model_best, formato="onnx")

    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO YOLO COMPLETADO")
    print(f"  Pesos en  : {PROJECT_DIR}/{RUN_NAME}/weights/")
    print(f"  best.pt   → mejor mAP@50 durante entrenamiento")
    print(f"  last.pt   → pesos de la última época")
    print("=" * 60)
