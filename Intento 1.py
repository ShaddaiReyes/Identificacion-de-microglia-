# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:55:25 2026

@author: shada
"""
import csv
from pathlib import Path

import numpy as np
from skimage import io, filters, feature
from skimage.segmentation import flood


def cargar_imagen(ruta, canal_interes=1):
    ruta_path = Path(ruta)
    
    # 1. Carga la imagen según su extensión
    if ruta_path.suffix.lower() == ".nd2":
        import nd2
        with nd2.ND2File(ruta) as f:
            img = f.asarray()
    else:
        img = io.imread(ruta)

    print(f"\n[INFO] Dimensiones originales de la imagen: {img.shape}")

    # 2. Manejo de dimensiones (CORREGIDO)
    if img.ndim == 3:
        # Si el último número es pequeño (ej. 3 canales), están al final (Y, X, Canales)
        if img.shape[-1] in [2, 3, 4, 5]:
            img = img[..., canal_interes]
        else:
            # Si no, asumimos que están al principio (Canales, Y, X)
            img = img[canal_interes]
            
    elif img.ndim == 4:
        img = img.max(axis=0) 
        if img.shape[-1] in [2, 3, 4, 5]:
            img = img[..., canal_interes]
        else:
            img = img[canal_interes]

    print(f"[INFO] Dimensiones después de extraer el canal {canal_interes}: {img.shape}")
    return img.astype(np.float64)


def encontrar_semillas_automaticas(img, distancia_min=15, umbral_relativo=0.4):
    suavizada = filters.gaussian(img, sigma=2)
    umbral = suavizada.max() * umbral_relativo
    coords = feature.peak_local_max(
        suavizada, min_distance=distancia_min, threshold_abs=umbral
    )
    return coords  


def crecer_region(img, semilla_yx, tolerancia):
    y, x = semilla_yx
    mascara = flood(img, (y, x), tolerance=tolerancia)
    return mascara


def segmentar(
    ruta_imagen,
    canal_interes=1,
    tolerancia=20,
    semillas_manuales=None,
    distancia_min=15,
    umbral_relativo=0.4,
    area_minima=30,
):
    img = cargar_imagen(ruta_imagen, canal_interes=canal_interes)
    
    # --- NUEVO: INVERTIR LA IMAGEN ---
    # Convertimos lo negro en blanco y lo blanco en negro para que 
    # la matemática detecte tus glías oscuras.
    img = img.max() - img 
    
    print(f"[INFO] Intensidad mínima: {img.min()} | Intensidad máxima: {img.max()}")
    
    
    if semillas_manuales:
        semillas = np.array(semillas_manuales)  
    else:
        semillas = encontrar_semillas_automaticas(
            img, distancia_min=distancia_min, umbral_relativo=umbral_relativo
        )

    if len(semillas) == 0:
        print("\n[ALERTA] No se encontraron semillas. Prueba bajando 'umbral_relativo'.")
        return img, None, []

    etiquetas = np.zeros(img.shape, dtype=np.int32)
    resultados = []

    for i, (y, x) in enumerate(semillas, start=1):
        y, x = int(y), int(x)
        if etiquetas[y, x] != 0:
            continue  

        mascara = crecer_region(img, (y, x), tolerancia)
        area = mascara.sum()

        if area < area_minima:
            continue  

        mascara_libre = mascara & (etiquetas == 0)
        etiquetas[mascara_libre] = i

        resultados.append(
            {
                "celula_id": i,
                "semilla_x": x,
                "semilla_y": y,
                "area_px": int(mascara_libre.sum()),
                "intensidad_media": float(img[mascara_libre].mean()),
            }
        )

    return img, etiquetas, resultados


def guardar_resultados(ruta_imagen, img, etiquetas, resultados, carpeta_salida):
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    base = Path(ruta_imagen).stem

    # 1. Imagen etiquetada (Forzamos la ruta a string con str() para evitar el OSError)
    ruta_etiquetas = str(carpeta_salida / f"{base}_etiquetas.tif")
    io.imsave(ruta_etiquetas, etiquetas.astype(np.uint16), check_contrast=False)

    # 2. Overlay de contornos
    from skimage.segmentation import find_boundaries
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contornos = find_boundaries(etiquetas, mode="outer")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray")
    overlay = np.zeros((*img.shape, 4))
    overlay[contornos] = [1, 0, 0, 1]  
    ax.imshow(overlay)
    for r in resultados:
        ax.text(r["semilla_x"], r["semilla_y"], str(r["celula_id"]),
                 color="yellow", fontsize=8)
    ax.axis("off")
    
    ruta_overlay = str(carpeta_salida / f"{base}_overlay.png")
    fig.savefig(ruta_overlay, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # 3. CSV con métricas
    ruta_csv = str(carpeta_salida / f"{base}_metricas.csv")
    with open(ruta_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["celula_id", "semilla_x", "semilla_y", "area_px", "intensidad_media"]
        )
        writer.writeheader()
        writer.writerows(resultados)

    return ruta_etiquetas, ruta_overlay, ruta_csv, None


# =============================================================================
#           CONFIGURACIÓN Y EJECUCIÓN DIRECTA EN SPYDER
# =============================================================================
if __name__ == "__main__":
    
    # 1. Tu archivo real (Asegúrate de que esté en la misma carpeta que el script)
    IMAGEN = r"C:\Users\shada\OneDrive\Desktop\Servicio\R1_5XVEH_40X_CA1_FOTO1_T2.nd2"
    
    # 2. Parámetros de la prueba inicial alta sensibilidad
    CANAL_INTERES = 1
    TOLERANCIA = 15.0
    DISTANCIA_MIN = 25
    UMBRAL_RELATIVO = 0.4  # Sensibilidad máxima
    AREA_MINIMA = 80 #Pixeles
    CARPETA_SALIDA = r"C:\Users\shada\OneDrive\Desktop\Servicio\resultados_segmentacion"
    
    # -------------------------------------------------------------------------
    print(f"\n--- INICIANDO PROCESAMIENTO DE: {IMAGEN} ---")
    
    try:
        img, etiquetas, resultados = segmentar(
            IMAGEN,
            canal_interes=CANAL_INTERES,
            tolerancia=TOLERANCIA,
            distancia_min=DISTANCIA_MIN,
            umbral_relativo=UMBRAL_RELATIVO,
            area_minima=AREA_MINIMA,
        )

        if resultados:
            print(f"\n¡Éxito! Se detectaron {len(resultados)} células.")
            rutas = guardar_resultados(IMAGEN, img, etiquetas, resultados, CARPETA_SALIDA)
            print("Revisa los archivos generados en la carpeta 'resultados_segmentacion'.")
            
    except FileNotFoundError:
        print(f"\n[ERROR CRÍTICO] Python no encuentra el archivo '{IMAGEN}'.")