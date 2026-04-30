import cv2
import numpy as np
import os
import albumentations as A

DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_ENTRADA = os.path.join(DIRECTORIO_BASE, "fotos_prueba/positive_photos")
CARPETA_SALIDA = os.path.join(DIRECTORIO_BASE, "dataset_con_lluvia")

transformacion_tormenta = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, -0.2), 
        contrast_limit=(-0.3, 0.0),    
        p=1.0 
    ),
    A.RandomRain(
        slant_lower=-10, slant_upper=10, 
        drop_length=20, drop_width=1,    
        drop_color=(200, 200, 200),      
        blur_value=3,                    
        brightness_coefficient=0.8,      
        rain_type='heavy',               
        p=1.0
    )
])

def inyectar_clima_adverso_aislado(ruta_entrada, ruta_salida):
    print(f"[INFO] Iniciando tormenta. Origen: {ruta_entrada}")
    print(f"[INFO] Destino seguro: {ruta_salida}\n")
    imagenes_generadas = 0

    for directorio_actual, subdirectorios, archivos in os.walk(ruta_entrada):
        
        ruta_relativa = os.path.relpath(directorio_actual, ruta_entrada)
        carpeta_destino_actual = os.path.join(ruta_salida, ruta_relativa)
        
        if not os.path.exists(carpeta_destino_actual):
            os.makedirs(carpeta_destino_actual)

        for archivo in archivos:
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_img_original = os.path.join(directorio_actual, archivo)
                
                img = cv2.imdecode(np.fromfile(ruta_img_original, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None: 
                    continue
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resultado = transformacion_tormenta(image=img_rgb)
                img_lluvia_bgr = cv2.cvtColor(resultado['image'], cv2.COLOR_RGB2BGR)
                
                nombre_sin_ext, ext = os.path.splitext(archivo)
                nuevo_nombre = f"{nombre_sin_ext}_lluvia{ext}"
                ruta_guardado = os.path.join(carpeta_destino_actual, nuevo_nombre)
                
                cv2.imencode(ext, img_lluvia_bgr)[1].tofile(ruta_guardado)
                
                imagenes_generadas += 1
                
        if archivos:
             print(f"  [OK] Carpeta '/{ruta_relativa}' procesada.")

    print(f"\n[EXITO] Se aislaron {imagenes_generadas} imágenes nuevas en: {ruta_salida}")

if __name__ == "__main__":
    if not os.path.exists(CARPETA_ENTRADA):
        print(f"[ERROR] No se encontró la carpeta de origen: {CARPETA_ENTRADA}")
    else:
        inyectar_clima_adverso_aislado(CARPETA_ENTRADA, CARPETA_SALIDA)