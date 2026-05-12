import os
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

# =============================================================================
# SECCIÓN 1: IMPORTACIONES Y CONFIGURACIÓN
# =============================================================================

DATASET_ROOT  = "dataset_dividido"

# Directorios de origen (donde están tus XML)
TRAIN_LBL_DIR = os.path.join(DATASET_ROOT, "Entrenamiento", "etiquetas")
VAL_LBL_DIR   = os.path.join(DATASET_ROOT, "Validacion",    "etiquetas")

# NUEVOS Directorios de destino (donde se crearán los TXT)
TRAIN_YOLO_DIR = os.path.join(DATASET_ROOT, "Entrenamiento", "labels")
VAL_YOLO_DIR   = os.path.join(DATASET_ROOT, "Validacion",    "labels")

CLASSES = [
    "person",       # 0
    "bicycle",      # 1
    "motorbike",    # 2
    "car",          # 3
    "bus",          # 4
    "truck",        # 5
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

LABEL_ALIASES = {
    "motorcycle": "motorbike",
}

YAML_OUTPUT = "data.yaml"

# =============================================================================
# SECCIÓN 2: FUNCIÓN DE CONVERSIÓN DE UN SOLO XML
# =============================================================================

def convert_xml_to_yolo(xml_path: str, output_txt_path: str) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size   = root.find("size")
    img_w  = float(size.find("width").text)
    img_h  = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip().lower()
        class_name = LABEL_ALIASES.get(class_name, class_name)

        if class_name not in CLASS_TO_IDX:
            continue

        class_idx = CLASS_TO_IDX[class_name]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width    = (xmax - xmin) / img_w
        height   = (ymax - ymin) / img_h

        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width    = min(max(width,    0.0), 1.0)
        height   = min(max(height,   0.0), 1.0)

        lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)

# =============================================================================
# SECCIÓN 3: CONVERSIÓN MASIVA (MODIFICADA PARA NUEVA CARPETA)
# =============================================================================

def convert_folder(xml_dir: str, output_dir: str, split_name: str) -> dict:
    """
    Lee XML de 'xml_dir' y guarda los TXT en 'output_dir'.
    """
    # Crear la carpeta de destino si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = list(Path(xml_dir).glob("*.xml"))
    total_objs = 0
    class_counts = {cls: 0 for cls in CLASSES}

    print(f"\n[{split_name}] Procesando {len(xml_files)} archivos XML...")
    print(f" 📂 Destino de etiquetas YOLO: {output_dir}")

    for xml_path in xml_files:
        # Definir la ruta de salida usando el nombre del archivo pero en la carpeta nueva
        txt_filename = xml_path.stem + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        n = convert_xml_to_yolo(str(xml_path), txt_path)
        total_objs += n

        if n > 0:
            tree = ET.parse(str(xml_path))
            for obj in tree.getroot().findall("object"):
                name = obj.find("name").text.strip().lower()
                name = LABEL_ALIASES.get(name, name)
                if name in class_counts:
                    class_counts[name] += 1

    print(f"  ✓ Conversión finalizada.")
    return {"files": len(xml_files), "objects": total_objs, "classes": class_counts}

# =============================================================================
# SECCIÓN 4: GENERACIÓN DE data.yaml
# =============================================================================

def generate_data_yaml(output_path: str):
    abs_root = str(Path(DATASET_ROOT).resolve())

    data = {
        "path":  abs_root,
        "train": os.path.join("Entrenamiento", "imagenes"),
        "val":   os.path.join("Validacion",    "imagenes"),
        "nc":    len(CLASSES),
        "names": CLASSES,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[data.yaml] Configuración actualizada.")

# =============================================================================
# SECCIÓN 5: PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  CONVERSIÓN XML (Pascal VOC) → TXT (YOLO) EN CARPETA SEPARADA")
    print("=" * 60)

    # Convertir entrenamiento (De: etiquetas -> A: labels)
    convert_folder(TRAIN_LBL_DIR, TRAIN_YOLO_DIR, "Entrenamiento")

    # Convertir validación (De: etiquetas -> A: labels)
    convert_folder(VAL_LBL_DIR, VAL_YOLO_DIR, "Validacion")

    # Generar data.yaml
    generate_data_yaml(YAML_OUTPUT)

    print("\n[✓] Proceso terminado. Las etiquetas originales XML siguen intactas.")