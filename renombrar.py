import os
import xml.etree.ElementTree as ET

DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__)) 
CARPETA_DATASET = os.path.join(DIRECTORIO_BASE, "fotos_prueba")

def renombrar_archivos_y_xml(ruta_base):
    print(f"[INFO] Iniciando renombrado en: {ruta_base}")

    for directorio_actual, subdirectorios, archivos in os.walk(ruta_base):
        if directorio_actual == ruta_base:
            continue
            
        nombre_carpeta = os.path.basename(directorio_actual)
        nombre_carpeta_limpio = nombre_carpeta.replace(" ", "_")
        
        contador = 1
        
        for archivo in archivos:
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_img_antigua = os.path.join(directorio_actual, archivo)
                nombre_sin_ext, ext = os.path.splitext(archivo)
                ruta_xml_antigua = os.path.join(directorio_actual, f"{nombre_sin_ext}.xml")
                
                nuevo_nombre_base = f"{nombre_carpeta_limpio}_{contador}"
                
                nuevo_nombre_img = f"{nuevo_nombre_base}{ext}"
                ruta_img_nueva = os.path.join(directorio_actual, nuevo_nombre_img)
                
                os.rename(ruta_img_antigua, ruta_img_nueva)
                
                if os.path.exists(ruta_xml_antigua):
                    ruta_xml_nueva = os.path.join(directorio_actual, f"{nuevo_nombre_base}.xml")
                    
                    tree = ET.parse(ruta_xml_antigua)
                    root = tree.getroot()
                    
                    nodo_filename = root.find('filename')
                    if nodo_filename is not None:
                        nodo_filename.text = nuevo_nombre_img
                    
                    nodo_path = root.find('path')
                    if nodo_path is not None:
                        nodo_path.text = ruta_img_nueva
                    
                    tree.write(ruta_xml_antigua, encoding="utf-8", xml_declaration=False)
                    
                    os.rename(ruta_xml_antigua, ruta_xml_nueva)
                    print(f"  [OK] Renombrado: {nuevo_nombre_img} (y su XML)")
                else:
                    print(f"  [OK] Renombrado: {nuevo_nombre_img} (Negativa, sin XML)")
                
                contador += 1

    print("\n[EXITO] ¡Todo el dataset ha sido renombrado y los XML actualizados de forma segura!")

if __name__ == "__main__":
    renombrar_archivos_y_xml(CARPETA_DATASET)