"""
=============================================================================
INFERENCIA EN TIEMPO REAL — SEMÁFORO INTELIGENTE (YOLOv8)
=============================================================================
Modos de uso:
    # Imagen individual
    python inferencia_yolo.py --source foto.jpg

    # Carpeta de imágenes
    python inferencia_yolo.py --source carpeta/

    # Video
    python inferencia_yolo.py --source video.mp4

    # Cámara en tiempo real (índice de cámara)
    python inferencia_yolo.py --source 0

    # Cambiar modelo o umbral
    python inferencia_yolo.py --source 0 --weights best.pt --conf 0.4
=============================================================================
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# --- Clases y colores BGR para OpenCV ---
CLASSES = ["person", "bicycle", "motorbike", "car", "bus", "truck"]

# Prioridad de cada clase para el semáforo (mayor = más prioridad)
# Lógica: bus y truck son vehículos grandes, tienen más peso en el semáforo
PRIORITY = {
    "person":    5,   # Peatones — máxima prioridad de seguridad
    "bicycle":   4,
    "motorbike": 3,
    "car":       2,
    "bus":       1,
    "truck":     1,
}

# Colores BGR para cada clase
COLORS = {
    "person":    (60,  60,  255),   # Rojo
    "bicycle":   (255, 160,  60),   # Azul claro
    "motorbike": (60, 160, 255),    # Naranja
    "car":       (60, 200,  60),    # Verde
    "bus":       (200,  60, 200),   # Magenta
    "truck":     (60, 220, 220),    # Amarillo
}


def calcular_fase_semaforo(detecciones: dict) -> str:
    """
    Lógica simplificada de semáforo inteligente basada en detecciones.

    Reglas:
      - Si hay peatones (person) → STOP (rojo para vehículos)
      - Si hay muchos vehículos (>5) → fase larga (verde extendido)
      - Si pocos vehículos (<3) → fase corta
      - Sin detecciones → amarillo intermitente

    En un sistema real esta lógica conectaría con el controlador del semáforo.
    """
    total_vehicles = sum(v for k, v in detecciones.items() if k != "person")
    persons = detecciones.get("person", 0)

    if persons > 0:
        return f"🔴 STOP — {persons} peatón(es) detectado(s)"
    elif total_vehicles > 5:
        return f"🟢 VERDE EXTENDIDO — {total_vehicles} vehículos"
    elif total_vehicles > 0:
        return f"🟢 VERDE NORMAL — {total_vehicles} vehículo(s)"
    else:
        return "🟡 LIBRE — Sin tráfico detectado"


def run(weights: str, source: str, conf: float, iou: float, show: bool):
    """
    Ejecuta inferencia con YOLOv8 sobre la fuente indicada.
    Soporta imagen, carpeta, video y cámara en tiempo real.
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    model  = YOLO(weights)
    model.to(device)

    print(f"[INFO] Modelo cargado: {weights}")
    print(f"[INFO] Fuente: {source}")
    print(f"[INFO] Conf: {conf} | IoU: {iou} | Device: {device}\n")

    # YOLOv8 maneja automáticamente imágenes, videos y streams
    results_gen = model.predict(
        source  = source,
        conf    = conf,
        iou     = iou,
        stream  = True,     # stream=True para procesar frame a frame (video/cámara)
        device  = device,
        verbose = False,
    )

    frame_idx = 0

    for result in results_gen:
        frame_idx += 1
        frame = result.orig_img.copy()   # Frame original BGR (numpy)

        # Conteo de detecciones por clase en este frame
        detecciones = {cls: 0 for cls in CLASSES}

        # Dibujar cada detección
        for box in result.boxes:
            cls_id   = int(box.cls.item())
            score    = box.conf.item()
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
            color    = COLORS.get(cls_name, (255, 255, 255))

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Etiqueta con fondo
            label = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

            detecciones[cls_name] += 1

        # Panel de conteo por clase (esquina superior izquierda)
        y_offset = 20
        for cls_name, count in detecciones.items():
            if count == 0:
                continue
            color = COLORS[cls_name]
            txt   = f"{cls_name}: {count}"
            cv2.putText(frame, txt, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 22

        # Fase del semáforo (parte inferior)
        fase = calcular_fase_semaforo(detecciones)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, fase, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Frame info
        cv2.putText(frame, f"Frame {frame_idx}", (w - 110, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if show:
            cv2.imshow("Semaforo Inteligente — YOLOv8", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Detenido por usuario (q)")
                break

    cv2.destroyAllWindows()
    print(f"[INFO] Procesados {frame_idx} frame(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferencia YOLOv8 — Semáforo Inteligente"
    )
    parser.add_argument(
        "--weights", default="runs_semaforo/yolov8m_semaforo_v13/weights/best.pt",
        help="Ruta al archivo de pesos .pt"
    )
    parser.add_argument(
        "--source",  default="0",
        help="Fuente: ruta de imagen/video, carpeta, o índice de cámara (0)"
    )
    parser.add_argument("--conf",  default=0.5,  type=float, help="Umbral de confianza")
    parser.add_argument("--iou",   default=0.45, type=float, help="IoU para NMS")
    parser.add_argument("--show",  action="store_true", default=True,
                        help="Mostrar ventana en tiempo real")
    args = parser.parse_args()

    run(
        weights = args.weights,
        source  = args.source,
        conf    = args.conf,
        iou     = args.iou,
        show    = args.show,
    )
