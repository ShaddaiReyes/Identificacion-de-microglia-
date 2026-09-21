# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:16:48 2026

@author: shada
"""

from pathlib import Path

import numpy as np
import pandas as pd
from skimage import feature, filters, io, measure
from skimage.segmentation import flood


def cargar_imagen(ruta, canal_interes=1):
    ruta_path = Path(ruta)

    if ruta_path.suffix.lower() == ".nd2":
        import nd2

        with nd2.ND2File(ruta) as f:
            img = f.asarray()
    else:
        img = io.imread(ruta)

    print(f"\n[INFO] Dimensiones originales: {img.shape}")

    if img.ndim == 3:
        if img.shape[-1] in [2, 3, 4, 5]:
            img = img[..., canal_interes]
        else:
            img = img[canal_interes]
    elif img.ndim == 4:
        img = img.max(axis=0)
        if img.shape[-1] in [2, 3, 4, 5]:
            img = img[..., canal_interes]
        else:
            img = img[canal_interes]

    print(f"[INFO] Dimensiones canal {canal_interes}: {img.shape}")
    return img.astype(np.float64)


def encontrar_semillas_automaticas(img, distancia_min=20, umbral_abs=None):
    suavizada = filters.gaussian(img, sigma=2)

    if umbral_abs is None:
        umbral_abs = np.median(suavizada) + (1.5 * np.std(suavizada))

    coords = feature.peak_local_max(
        suavizada, min_distance=distancia_min, threshold_abs=umbral_abs
    )
    return coords


def crecer_region(img, semilla_yx, tolerancia):
    y, x = semilla_yx
    mascara = flood(img, (y, x), tolerance=tolerancia)
    return mascara


def segmentar_con_regionprops(
    ruta_imagen,
    canal_interes=1,
    tolerancia=None,
    semillas_manuales=None,
    distancia_min=20,
    area_minima=80,
):
    img_orig = cargar_imagen(ruta_imagen, canal_interes=canal_interes)

    # Invertir para que las glías sean picos brillantes
    img_inv = img_orig.max() - img_orig

    # Cálculo adaptativo si no se proveen valores
    desviacion = np.std(img_inv)
    umbral_abs = np.median(img_inv) + (1.2 * desviacion)

    if tolerancia is None:
        tolerancia = desviacion * 0.75

    if semillas_manuales:
        semillas = np.array(semillas_manuales)
    else:
        semillas = encontrar_semillas_automaticas(
            img_inv, distancia_min=distancia_min, umbral_abs=umbral_abs
        )

    if len(semillas) == 0:
        return img_orig, None, []

    etiquetas = np.zeros(img_inv.shape, dtype=np.int32)

    # Asignación consecutiva de IDs
    id_celula = 1
    for y, x in semillas:
        y, x = int(y), int(x)
        if etiquetas[y, x] != 0:
            continue

        mascara = crecer_region(img_inv, (y, x), tolerancia)
        mascara_libre = mascara & (etiquetas == 0)

        if mascara_libre.sum() < area_minima:
            continue

        etiquetas[mascara_libre] = id_celula
        id_celula += 1

    propiedades = [
        "label",
        "centroid",
        "area",
        "perimeter",
        "eccentricity",
        "solidity",
        "equivalent_diameter_area",
        "mean_intensity",
    ]

    tabla_props = measure.regionprops_table(
        etiquetas, intensity_image=img_orig, properties=propiedades
    )

    df = pd.DataFrame(tabla_props)
    df = df.rename(
        columns={
            "label": "celula_id",
            "centroid-0": "semilla_y",
            "centroid-1": "semilla_x",
            "area": "area_px",
            "perimeter": "perimetro_px",
            "eccentricity": "excentricidad",
            "solidity": "solidez",
            "equivalent_diameter_area": "diametro_equiv",
            "mean_intensity": "intensidad_media",
        }
    )

    resultados = df.to_dict(orient="records")
    return img_orig, etiquetas, resultados


def guardar_resultados(ruta_imagen, img, etiquetas, resultados, carpeta_salida):
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    base = Path(ruta_imagen).stem

    # 1. Imagen etiquetada
    ruta_etiquetas = str(carpeta_salida / f"{base}_etiquetas.tif")
    io.imsave(
        ruta_etiquetas, etiquetas.astype(np.uint16), check_contrast=False
    )

    # 2. Overlay visual: Fondo claro, Glías oscuras con alto contraste
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Normalizar y estirar contraste
    p1, p99 = np.percentile(img, (1, 99))
    img_norm = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        img_norm, cmap="gray"
    )  # Muestra el fondo blanco con glías oscuras
    ax.axis("off")

    ruta_overlay = str(carpeta_salida / f"{base}_overlay.png")
    fig.savefig(ruta_overlay, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # 3. CSV
    ruta_csv = str(carpeta_salida / f"{base}_metricas.csv")
    df = pd.DataFrame(resultados)
    df.to_csv(ruta_csv, index=False)

    return ruta_etiquetas, ruta_overlay, ruta_csv, None


if __name__ == "__main__":
    IMAGEN = (
        r"C:\Users\shada\OneDrive\Desktop\Servicio\R1_5XVEH_40X_CA1_FOTO1_T2.nd2"
    )
    CARPETA_SALIDA = (
        r"C:\Users\shada\OneDrive\Desktop\Servicio\resultados_segmentacion"
    )

    try:
        img, etiquetas, resultados = segmentar_con_regionprops(
            IMAGEN,
            canal_interes=1,
            tolerancia=None,  # Automático
            distancia_min=25,
            area_minima=80,
        )

        if resultados:
            print(f"\nSe detectaron {len(resultados)} células.")
            rutas = guardar_resultados(
                IMAGEN, img, etiquetas, resultados, CARPETA_SALIDA
            )
            print("Archivos generados en 'resultados_segmentacion'.")

    except FileNotFoundError:
        print(f"\n[ERROR CRÍTICO] Archivo no encontrado: '{IMAGEN}'")    except FileNotFoundError:
        print(f"\n[ERROR CRÍTICO] Archivo no encontrado: '{IMAGEN}'")
