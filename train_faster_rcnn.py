"""
=============================================================================
SISTEMA DE SEMÁFORO INTELIGENTE - ENTRENAMIENTO FASTER R-CNN
=============================================================================
Detecta: bicycle, motorbike, bus, truck, person, car
Estructura esperada:
    dataset_dividido/
        Entrenamiento/
            imagenes/
            etiquetas/       <- archivos .xml formato Pascal VOC
        Validacion/
            imagenes/
            etiquetas/
=============================================================================
"""

# =============================================================================
# SECCIÓN 1: IMPORTACIONES
# Se importan todas las librerías necesarias para entrenamiento, manejo de
# datos, modelo y visualización de métricas.
# =============================================================================
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import json
import random
from pathlib import Path

# =============================================================================
# SECCIÓN 2: CONFIGURACIÓN GENERAL
# Define todas las rutas, hiperparámetros y clases del proyecto.
# Modifica esta sección para adaptar el script a tu entorno.
# =============================================================================

# --- Rutas del dataset ---
DATASET_ROOT    = "dataset_dividido"
TRAIN_IMG_DIR   = os.path.join(DATASET_ROOT, "Entrenamiento", "imagenes")
TRAIN_LBL_DIR   = os.path.join(DATASET_ROOT, "Entrenamiento", "etiquetas")
VAL_IMG_DIR     = os.path.join(DATASET_ROOT, "Validacion",    "imagenes")
VAL_LBL_DIR     = os.path.join(DATASET_ROOT, "Validacion",    "etiquetas")

# --- Directorio de salida para checkpoints y métricas ---
OUTPUT_DIR = "output_faster_rcnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Clases del proyecto (índice 0 reservado para fondo/background) ---
CLASSES = [
    "__background__",   # 0 - requerido por Faster R-CNN
    "person",           # 1
    "bicycle",          # 2 - persona en bicicleta
    "motorbike",        # 3 - persona en moto
    "car",              # 4
    "bus",              # 5
    "truck",            # 6
]
NUM_CLASSES = len(CLASSES)          # 7 (incluyendo background)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# --- Hiperparámetros de entrenamiento ---
NUM_EPOCHS      = 20        # Épocas totales de entrenamiento
BATCH_SIZE      = 4         # Imágenes por batch (reduce si hay OOM)
LEARNING_RATE   = 0.005     # Tasa de aprendizaje inicial
MOMENTUM        = 0.9       # Momentum para SGD
WEIGHT_DECAY    = 0.0005    # Regularización L2
LR_STEP_SIZE    = 7         # Reducir LR cada N épocas
LR_GAMMA        = 0.1       # Factor de reducción del LR
NUM_WORKERS     = 2         # Hilos para carga de datos (0 en Windows)
SCORE_THRESHOLD = 0.5       # Umbral mínimo de confianza en inferencia

# --- Dispositivo de cómputo ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# SECCIÓN 3: DATASET PERSONALIZADO
# Clase que carga imágenes y sus anotaciones XML (formato Pascal VOC).
# __getitem__ devuelve (imagen_tensor, target_dict) compatible con torchvision.
# =============================================================================

class SemaforoDataset(Dataset):
    """
    Dataset para detección de objetos en intersecciones viales.
    Lee imágenes (.jpg, .png) y anotaciones Pascal VOC (.xml).
    """

    # Extensiones de imagen válidas
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, img_dir: str, lbl_dir: str, transforms=None):
        """
        Args:
            img_dir   : Ruta a la carpeta de imágenes.
            lbl_dir   : Ruta a la carpeta de etiquetas XML.
            transforms: Transformaciones a aplicar a imagen y boxes.
        """
        self.img_dir    = img_dir
        self.lbl_dir    = lbl_dir
        self.transforms = transforms

        # Recolectar todos los archivos de imagen disponibles
        self.imgs = sorted([
            f for f in os.listdir(img_dir)
            if Path(f).suffix.lower() in self.IMG_EXTS
        ])

        # Filtrar imágenes que tienen su XML correspondiente
        self.imgs = [
            f for f in self.imgs
            if os.path.exists(
                os.path.join(lbl_dir, Path(f).stem + ".xml")
            )
        ]

        print(f"[Dataset] {img_dir} → {len(self.imgs)} imágenes con etiqueta")

    def __len__(self):
        return len(self.imgs)

    def _parse_xml(self, xml_path: str):
        """
        Parsea un archivo XML Pascal VOC y extrae boxes y etiquetas.
        Retorna:
            boxes  : lista de [xmin, ymin, xmax, ymax]
            labels : lista de índices de clase (int)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes  = []
        labels = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip().lower()

            # Ignorar clases no definidas en el proyecto
            if class_name not in CLASS_TO_IDX:
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Validar que el bounding box sea válido (xmax > xmin, etc.)
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[class_name])

        return boxes, labels

    def __getitem__(self, idx):
        """
        Retorna:
            image  : Tensor float32 [3, H, W] normalizado [0,1]
            target : dict con 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        img_name  = self.imgs[idx]
        img_path  = os.path.join(self.img_dir, img_name)
        xml_path  = os.path.join(self.lbl_dir, Path(img_name).stem + ".xml")

        # Cargar imagen en RGB
        image = Image.open(img_path).convert("RGB")

        # Parsear anotaciones
        boxes, labels = self._parse_xml(xml_path)

        # Convertir a tensores
        if len(boxes) > 0:
            boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Imagen sin anotaciones válidas → tensores vacíos
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        # Calcular área de cada bounding box
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) \
               if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  torch.zeros((len(labels),), dtype=torch.int64),
        }

        # Aplicar transformaciones (convierte PIL → Tensor)
        if self.transforms:
            image = self.transforms(image)

        return image, target


# =============================================================================
# SECCIÓN 4: TRANSFORMACIONES / DATA AUGMENTATION
# Las transformaciones de entrenamiento incluyen aumentación de datos para
# mejorar la generalización. Validación solo convierte a tensor.
# =============================================================================

def get_train_transforms():
    """
    Transformaciones de entrenamiento:
    - ColorJitter: variaciones aleatorias de brillo, contraste, saturación
    - RandomHorizontalFlip: volteo horizontal (p=0.5)
    - ToTensor: convierte PIL Image a Tensor [0,1]
    NOTA: Faster R-CNN de torchvision acepta imágenes sin normalización
    adicional (la backbone ResNet50 normaliza internamente con ImageNet stats).
    """
    return T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ])

def get_val_transforms():
    """
    Transformaciones de validación:
    Solo se convierte la imagen a Tensor sin augmentación.
    """
    return T.Compose([
        T.ToTensor(),
    ])


# =============================================================================
# SECCIÓN 5: FUNCIÓN COLLATE_FN
# Faster R-CNN requiere que cada item del batch sea una tupla (imagen, target).
# Esta función evita el apilado automático de PyTorch (que fallaría porque
# los targets tienen distinto número de objetos por imagen).
# =============================================================================

def collate_fn(batch):
    """Agrupa un batch como lista de tuplas (imagen, target)."""
    return tuple(zip(*batch))


# =============================================================================
# SECCIÓN 6: CONSTRUCCIÓN DEL MODELO
# Se carga Faster R-CNN con backbone ResNet-50 + FPN preentrenado en COCO.
# El clasificador final se reemplaza por uno con NUM_CLASSES salidas.
# =============================================================================

def build_model(num_classes: int):
    """
    Construye Faster R-CNN con:
    - Backbone: ResNet-50 con Feature Pyramid Network (FPN)
    - Pesos iniciales: COCO 2017 (transfer learning)
    - Cabeza de clasificación: ajustada a num_classes

    Transfer Learning:
    - Las capas convolucionales ya aprendieron características visuales ricas.
    - Solo se reentrena la cabeza de clasificación (box_predictor).
    - Esto reduce tiempo de entrenamiento y mejora rendimiento con pocos datos.
    """
    # Cargar modelo preentrenado en COCO
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Obtener el número de características de entrada del clasificador actual
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Reemplazar la cabeza de clasificación con una nueva para nuestras clases
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# =============================================================================
# SECCIÓN 7: FUNCIÓN DE ENTRENAMIENTO POR ÉPOCA
# Ejecuta un ciclo completo de forward + backward + optimización sobre
# todos los batches del conjunto de entrenamiento.
# =============================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Entrena el modelo durante una época completa.

    Args:
        model       : Modelo Faster R-CNN
        optimizer   : Optimizador (SGD)
        data_loader : DataLoader de entrenamiento
        device      : CPU o CUDA
        epoch       : Número de época actual (para logging)

    Returns:
        avg_loss : Pérdida promedio de la época
    """
    model.train()   # Activa BatchNorm y Dropout en modo entrenamiento

    total_loss = 0.0
    num_batches = len(data_loader)

    for batch_idx, (images, targets) in enumerate(data_loader):
        # Mover datos al dispositivo (GPU/CPU)
        images  = [img.to(device)   for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # ── Forward pass ──────────────────────────────────────────────────
        # En modo train, Faster R-CNN retorna un dict de pérdidas:
        #   loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
        loss_dict = model(images, targets)

        # Suma todas las pérdidas parciales
        losses = sum(loss for loss in loss_dict.values())

        # ── Backward pass ─────────────────────────────────────────────────
        optimizer.zero_grad()   # Limpiar gradientes del batch anterior
        losses.backward()       # Calcular gradientes
        optimizer.step()        # Actualizar pesos

        total_loss += losses.item()

        # Log de progreso cada 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            loss_strs = " | ".join(
                f"{k}: {v.item():.4f}" for k, v in loss_dict.items()
            )
            print(
                f"  Época [{epoch}] Batch [{batch_idx+1}/{num_batches}] "
                f"Loss total: {losses.item():.4f} → {loss_strs}"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


# =============================================================================
# SECCIÓN 8: FUNCIÓN DE VALIDACIÓN
# Calcula la pérdida sobre el conjunto de validación sin actualizar pesos.
# Se usa torch.no_grad() para ahorrar memoria y acelerar el proceso.
# =============================================================================

@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evalúa el modelo en el conjunto de validación.

    NOTA: Faster R-CNN solo devuelve pérdidas en modo train().
    Para obtener la pérdida de validación se activa train() temporalmente
    pero con torch.no_grad() para NO acumular gradientes.

    Returns:
        avg_val_loss : Pérdida promedio de validación
    """
    # Activar modo entrenamiento para obtener pérdidas, pero sin gradientes
    model.train()

    total_loss = 0.0
    num_batches = len(data_loader)

    for images, targets in data_loader:
        images  = [img.to(device)   for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    avg_val_loss = total_loss / num_batches
    return avg_val_loss


# =============================================================================
# SECCIÓN 9: GUARDADO DE CHECKPOINTS
# Guarda el estado del modelo, optimizador y métricas al final de cada época.
# Permite retomar el entrenamiento desde cualquier punto si se interrumpe.
# =============================================================================

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """
    Guarda un checkpoint completo del estado del entrenamiento.

    Args:
        model      : Modelo con pesos actuales
        optimizer  : Estado del optimizador (momentos, lr, etc.)
        epoch      : Época actual
        train_loss : Pérdida de entrenamiento de la época
        val_loss   : Pérdida de validación de la época
        path       : Ruta de guardado del archivo .pth
    """
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "train_loss":  train_loss,
        "val_loss":    val_loss,
        "classes":     CLASSES,
        "num_classes": NUM_CLASSES,
    }, path)
    print(f"  [Checkpoint] Guardado en: {path}")


# =============================================================================
# SECCIÓN 10: GRAFICACIÓN DE MÉTRICAS
# Genera y guarda la gráfica de pérdida entrenamiento vs validación por época.
# =============================================================================

def plot_losses(train_losses, val_losses, output_dir):
    """
    Genera una gráfica comparativa de pérdida de entrenamiento y validación.

    Args:
        train_losses : Lista de pérdidas de entrenamiento por época
        val_losses   : Lista de pérdidas de validación por época
        output_dir   : Carpeta donde se guardará la imagen
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-o", label="Pérdida Entrenamiento", linewidth=2)
    plt.plot(epochs, val_losses,   "r-o", label="Pérdida Validación",    linewidth=2)
    plt.xlabel("Época",   fontsize=13)
    plt.ylabel("Pérdida", fontsize=13)
    plt.title("Faster R-CNN — Semáforo Inteligente\nPérdida por Época", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Gráfica] Guardada en: {path}")


# =============================================================================
# SECCIÓN 11: INFERENCIA DE EJEMPLO
# Realiza una predicción sobre imágenes de validación y las visualiza con
# los bounding boxes predichos. Útil para verificar calidad del modelo.
# =============================================================================

@torch.no_grad()
def visualize_predictions(model, val_dataset, device, output_dir,
                           num_samples=4, score_thresh=SCORE_THRESHOLD):
    """
    Genera imágenes con las predicciones del modelo superpuestas.

    Args:
        model       : Modelo entrenado en modo eval()
        val_dataset : Dataset de validación
        device      : CPU o CUDA
        output_dir  : Carpeta de salida
        num_samples : Número de imágenes a visualizar
        score_thresh: Umbral de confianza mínimo para mostrar detecciones
    """
    model.eval()

    # Paleta de colores para cada clase (excluye background)
    colors = {
        "person":    "#FF4444",
        "bicycle":   "#44AAFF",
        "motorbike": "#FF8800",
        "car":       "#44FF44",
        "bus":       "#FF44FF",
        "truck":     "#FFFF44",
    }

    indices = random.sample(range(len(val_dataset)),
                            min(num_samples, len(val_dataset)))

    fig, axes = plt.subplots(1, len(indices), figsize=(6 * len(indices), 6))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        image_tensor, target = val_dataset[idx]

        # Forward pass — modelo en eval() retorna predicciones
        preds = model([image_tensor.to(device)])[0]

        # Convertir tensor a imagen numpy para matplotlib
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)
        ax.axis("off")

        # Dibujar cada detección que supere el umbral de confianza
        for box, label, score in zip(
            preds["boxes"], preds["labels"], preds["scores"]
        ):
            if score < score_thresh:
                continue

            class_name = CLASSES[label.item()]
            color      = colors.get(class_name, "#FFFFFF")
            x1, y1, x2, y2 = box.cpu().numpy()

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 4,
                f"{class_name} {score:.2f}",
                color=color, fontsize=8, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1)
            )

    plt.suptitle("Predicciones — Validación", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Visualización] Guardada en: {path}")


# =============================================================================
# SECCIÓN 12: BUCLE PRINCIPAL DE ENTRENAMIENTO
# Orquesta todo el proceso: carga datos, construye modelo, entrena,
# valida, guarda checkpoints y al final genera métricas y visualizaciones.
# =============================================================================

def main():
    print("=" * 65)
    print("  FASTER R-CNN — SISTEMA DE SEMÁFORO INTELIGENTE")
    print(f"  Clases: {CLASSES[1:]}")
    print(f"  Épocas: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print("=" * 65)

    # ── 12.1 Crear Datasets ───────────────────────────────────────────────
    train_dataset = SemaforoDataset(
        TRAIN_IMG_DIR, TRAIN_LBL_DIR,
        transforms=get_train_transforms()
    )
    val_dataset = SemaforoDataset(
        VAL_IMG_DIR, VAL_LBL_DIR,
        transforms=get_val_transforms()
    )

    # ── 12.2 Crear DataLoaders ────────────────────────────────────────────
    # shuffle=True en entrenamiento para que el modelo no memorice el orden
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # ── 12.3 Construir modelo y moverlo al dispositivo ────────────────────
    model = build_model(NUM_CLASSES)
    model.to(DEVICE)
    print(f"\n[Modelo] Faster R-CNN ResNet50-FPN con {NUM_CLASSES} clases cargado")

    # ── 12.4 Optimizador ─────────────────────────────────────────────────
    # SGD con momentum es el estándar para entrenar detectores de objetos.
    # Solo se optimizan los parámetros que requieren gradiente.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # ── 12.5 Scheduler de tasa de aprendizaje ────────────────────────────
    # Reduce el LR por un factor GAMMA cada LR_STEP_SIZE épocas.
    # Permite convergencia fina después de una fase de exploración amplia.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )

    # ── 12.6 Historial de métricas ────────────────────────────────────────
    train_losses = []
    val_losses   = []
    best_val_loss = float("inf")
    history_path  = os.path.join(OUTPUT_DIR, "training_history.json")

    # ── 12.7 Ciclo de entrenamiento ───────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  ÉPOCA {epoch}/{NUM_EPOCHS}   LR actual: {lr_scheduler.get_last_lr()}")
        print(f"{'─'*60}")

        # Entrenamiento
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)

        # Validación
        val_loss = evaluate(model, val_loader, DEVICE)

        # Actualizar scheduler
        lr_scheduler.step()

        # Registrar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        print(
            f"\n  Resumen Época {epoch}: "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Tiempo={elapsed:.1f}s"
        )

        # Guardar checkpoint de la época actual
        epoch_ckpt = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch:02d}.pth")
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, epoch_ckpt)

        # Guardar el mejor modelo basado en pérdida de validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_path)
            print(f"  ★ Nuevo mejor modelo guardado (val_loss={val_loss:.4f})")

        # Guardar historial de métricas en JSON (por si se interrumpe)
        with open(history_path, "w") as f:
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # ── 12.8 Post-entrenamiento ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"  Mejor Val Loss: {best_val_loss:.4f}")
    print(f"  Checkpoints y modelo final en: {OUTPUT_DIR}/")
    print(f"{'='*65}")

    # Graficar curvas de pérdida
    plot_losses(train_losses, val_losses, OUTPUT_DIR)

    # Visualizar predicciones de ejemplo con el mejor modelo cargado
    best_ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"),
                           map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])
    visualize_predictions(model, val_dataset, DEVICE, OUTPUT_DIR)

    # Guardar modelo final exportable (solo pesos)
    final_weights_path = os.path.join(OUTPUT_DIR, "faster_rcnn_semaforo_final.pth")
    torch.save(model.state_dict(), final_weights_path)
    print(f"  [Final] Pesos del modelo guardados en: {final_weights_path}")


# =============================================================================
# SECCIÓN 13: PUNTO DE ENTRADA
# Ejecuta el entrenamiento solo cuando se llama directamente este script.
# =============================================================================

if __name__ == "__main__":
    # Semilla para reproducibilidad (mismas inicializaciones aleatorias)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    main()
