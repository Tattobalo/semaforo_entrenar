"""
=============================================================================
INFERENCIA — SISTEMA DE SEMÁFORO INTELIGENTE
Carga el modelo entrenado y realiza detección sobre una imagen o carpeta.
=============================================================================
Uso:
    python inferencia.py --img ruta/a/imagen.jpg
    python inferencia.py --folder ruta/carpeta/  --out resultados/
=============================================================================
"""

import os, argparse, torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from train_faster_rcnn import build_model, CLASSES, NUM_CLASSES, DEVICE

COLORS = {
    "person":    "#FF4444",
    "bicycle":   "#44AAFF",
    "motorbike": "#FF8800",
    "car":       "#44FF44",
    "bus":       "#FF44FF",
    "truck":     "#FFFF44",
}

def load_model(weights_path: str):
    model = build_model(NUM_CLASSES)
    state = torch.load(weights_path, map_location=DEVICE)
    # Compatible con checkpoint completo o solo pesos
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def predict(model, image_path: str, score_thresh=0.5):
    img = Image.open(image_path).convert("RGB")
    tensor = T.ToTensor()(img).unsqueeze(0).to(DEVICE)
    preds  = model(tensor)[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img)
    ax.axis("off")

    detections = 0
    for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        if score < score_thresh:
            continue
        detections += 1
        cls   = CLASSES[label.item()]
        color = COLORS.get(cls, "#FFFFFF")
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1-4, f"{cls} {score:.2f}",
                color=color, fontsize=9, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1))

    plt.title(f"{Path(image_path).name}  |  Detecciones: {detections}")
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="output_faster_rcnn/best_model.pth")
    parser.add_argument("--img",    default=None, help="Ruta a imagen individual")
    parser.add_argument("--folder", default=None, help="Ruta a carpeta de imágenes")
    parser.add_argument("--out",    default="inferencias/", help="Carpeta de salida")
    parser.add_argument("--thresh", default=0.5, type=float)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.weights)
    print(f"[Inferencia] Modelo cargado desde {args.weights}")

    imgs = []
    if args.img:
        imgs = [args.img]
    elif args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = [str(p) for p in Path(args.folder).iterdir()
                if p.suffix.lower() in exts]

    for img_path in imgs:
        fig = predict(model, img_path, score_thresh=args.thresh)
        out_path = os.path.join(args.out, Path(img_path).stem + "_pred.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {out_path}")

    print(f"\n[Listo] {len(imgs)} imagen(es) procesadas en '{args.out}'")
