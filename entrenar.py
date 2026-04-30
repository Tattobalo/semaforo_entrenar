import cv2
import numpy as np
import os

# --- 1. CONFIGURACIÓN DEL MODELO ---
# Ajusta estas rutas a donde tienes tus archivos de MobileNet-SSD
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_PROTOTXT = os.path.join(DIRECTORIO_ACTUAL, "MobileNetSSD_deploy.prototxt")
RUTA_PESOS = os.path.join(DIRECTORIO_ACTUAL, "MobileNetSSD_deploy.caffemodel")

CLASES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

print("[INFO] Cargando modelo SSD para auto-etiquetado...")
net = cv2.dnn.readNetFromCaffe(RUTA_PROTOTXT, RUTA_PESOS)

# --- 2. GENERADOR DE XML (PASCAL VOC) ---
def generar_xml_pascal_voc(ruta_imagen, ancho, alto, detecciones_validas, carpeta_salida):
    nombre_archivo = os.path.basename(ruta_imagen)
    nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
    
    xml = f"""<annotation>
    <folder>dataset</folder>
    <filename>{nombre_archivo}</filename>
    <path>{ruta_imagen}</path>
    <source><database>SemaforoInteligente</database></source>
    <size>
        <width>{ancho}</width>
        <height>{alto}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
"""
    for det in detecciones_validas:
        xml += f"""    <object>
        <name>{det['clase']}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{det['xmin']}</xmin>
            <ymin>{det['ymin']}</ymin>
            <xmax>{det['xmax']}</xmax>
            <ymax>{det['ymax']}</ymax>
        </bndbox>
    </object>
"""
    xml += "</annotation>"
    
    ruta_xml = os.path.join(carpeta_salida, f"{nombre_sin_ext}.xml")
    with open(ruta_xml, "w", encoding="utf-8") as f:
        f.write(xml)

# --- 3. MOTOR DE INFERENCIA ---
def auto_etiquetar(carpeta_imagenes, carpeta_salida_xml, umbral=0.2):
    if not os.path.exists(carpeta_salida_xml):
        os.makedirs(carpeta_salida_xml)

    imagenes_procesadas = 0
    
    for archivo in os.listdir(carpeta_imagenes):
        if not archivo.lower().endswith(('.png', '.jpg', '.jpeg')): 
            continue
            
        ruta_img = os.path.join(carpeta_imagenes, archivo)
        
        # Lectura segura para rutas con caracteres especiales
        img = cv2.imdecode(np.fromfile(ruta_img, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: 
            continue
            
        (alto, ancho) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        
        net.setInput(blob)
        detecciones = net.forward()

        objetos_en_foto = []
        
        for i in np.arange(0, detecciones.shape[2]):
            confianza = detecciones[0, 0, i, 2]
            
            if confianza > umbral:
                idx_clase = int(detecciones[0, 0, i, 1])
                nombre_clase = CLASES[idx_clase]
                
                # Solo guardamos las clases del semáforo inteligente
                if nombre_clase in ["car", "bus", "motorbike", "bicycle", "person"]:
                    caja = detecciones[0, 0, i, 3:7] * np.array([ancho, alto, ancho, alto])
                    (startX, startY, endX, endY) = caja.astype("int")
                    
                    # Limitar coordenadas a los bordes de la imagen
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(ancho, endX), min(alto, endY)
                    
                    objetos_en_foto.append({
                        'clase': nombre_clase, 
                        'xmin': startX, 'ymin': startY, 
                        'xmax': endX, 'ymax': endY
                    })
        
        # Generar XML aunque se detectó al menos un objeto relevante
        generar_xml_pascal_voc(ruta_img, ancho, alto, objetos_en_foto, carpeta_salida_xml)
        imagenes_procesadas += 1
        print(f"[OK] Generado: {archivo.split('.')[0]}.xml con {len(objetos_en_foto)} objetos.")

    print(f"\n[EXITO] Se generaron etiquetas para {imagenes_procesadas} imágenes en: {carpeta_salida_xml}")

# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    # 1. Crea una carpeta llamada "fotos_prueba" y mete ahí unas 5 o 10 fotos
    CARPETA_ENTRADA = os.path.join(DIRECTORIO_ACTUAL, "dataset_prueba1")
    
    # 2. El script creará esta carpeta y guardará los XML ahí
    CARPETA_SALIDA = os.path.join(DIRECTORIO_ACTUAL, "etiquetas_generadas")
    
    # Asegurarnos de que la carpeta de entrada exista para evitar errores
    if not os.path.exists(CARPETA_ENTRADA):
        print(f"[ERROR] Por favor crea la carpeta '{CARPETA_ENTRADA}' y pon algunas fotos dentro.")
    else:
        print(f"[INFO] Analizando imágenes en: {CARPETA_ENTRADA}")
        auto_etiquetar(CARPETA_ENTRADA, CARPETA_SALIDA, umbral=0.2)