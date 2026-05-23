"""
analisis_frame.py
=================
Archivo hijo que integra los tres notebooks del grupo:
  - Preprocesamiento con NL-means y Gaussiano    (de Lu)
  - Skull stripping + segmentación por solidez   (de Nato)
  - Overlay rojo + localización anatómica AAL    (de Santino)

Uso:
    python analisis_frame.py

Ctrl+C para cortar ejecución.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import filters, morphology
from skimage.morphology import opening, closing, footprint_rectangle, erosion
from skimage.measure import label, regionprops
from skimage.restoration import denoise_nl_means
from nilearn import datasets
from nilearn.image import load_img

# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — modificar acá si cambia el archivo o el corte
# ══════════════════════════════════════════════════════════════
RUTA_ARCHIVO = "./data/sub-KA02/anat/sub-KA02_run-02_T1w.nii.gz"
FRAME        = 200   # corte axial a analizar

# ══════════════════════════════════════════════════════════════
#  ATLAS HARVARD-OXFORD
# ══════════════════════════════════════════════════════════════
print("Cargando atlas Harvard-Oxford...")
_atlas      = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
_atlas_img  = load_img(_atlas.maps)
_atlas_data = _atlas_img.get_fdata()
_atlas_aff  = _atlas_img.affine
AAL_LABELS  = {i: name for i, name in enumerate(_atlas.labels)}

CRITICAL_KEYWORDS = [
    "Precentral", "Broca", "Superior Temporal", "Cerebellum",
    "Hippocampus", "Putamen", "Caudate", "Pallidum",
    "Thalamus", "Postcentral", "Calcarine",
]

def is_critical(label_name: str) -> bool:
    return any(kw.lower() in label_name.lower() for kw in CRITICAL_KEYWORDS)

def localizar(centroid_vox, affine):
    cy, cx = centroid_vox   # regionprops da (row, col)
    
    # Dimensiones del volumen del paciente
    sx, sy, sz = data.shape[:3]
    
    # Dimensiones del atlas Harvard-Oxford (91x109x91 a 2mm)
    ax_dim, ay_dim, az_dim = _atlas_data.shape[:3]
    
    # Escalar coordenadas del paciente al espacio del atlas por proporción
    ax = int(round(cx / sx * ax_dim))
    ay = int(round(cy / sy * ay_dim))
    az = int(round(FRAME / sz * az_dim))
    
    # Verificar límites
    if not (0 <= ax < ax_dim and 0 <= ay < ay_dim and 0 <= az < az_dim):
        return "Fuera del espacio del atlas", False

    label_idx = int(_atlas_data[ax, ay, az])
    
    if label_idx == 0:
        # Buscar en vóxeles vecinos
        for dx, dy, dz in [(3,0,0),(-3,0,0),(0,3,0),(0,-3,0),(0,0,3),(0,0,-3)]:
            nx, ny, nz = ax+dx, ay+dy, az+dz
            if 0<=nx<ax_dim and 0<=ny<ay_dim and 0<=nz<az_dim:
                idx2 = int(_atlas_data[nx, ny, nz])
                if idx2 > 0:
                    nombre = AAL_LABELS.get(idx2, f"Región {idx2}")
                    return nombre, is_critical(nombre)
        return "Zona no especificada", False

    nombre = AAL_LABELS.get(label_idx, f"Región {label_idx}")
    return nombre, is_critical(nombre)


# ══════════════════════════════════════════════════════════════
#  PASO 1 — CARGA
# ══════════════════════════════════════════════════════════════
print(f"\nCargando: {RUTA_ARCHIVO}  |  frame {FRAME}")
img    = nib.load(RUTA_ARCHIVO)
data   = img.get_fdata()
affine = img.affine
frame  = data[:, :, FRAME]
print(f"  Dimensiones volumen: {data.shape}")
print(f"  Rango frame: [{frame.min():.1f}, {frame.max():.1f}]")


# ══════════════════════════════════════════════════════════════
#  PASO 2 — PREPROCESAMIENTO  (de Lu)
# ══════════════════════════════════════════════════════════════
print("\n[1/4] Preprocesando (NL-means + Gaussiano)...")

# Normalizar a [0,1]
f_min, f_max = frame.min(), frame.max()
image = (frame - f_min) / (f_max - f_min)

# Estimar sigma en esquina sin anatomía
centro      = image[image.shape[0]//4 : image.shape[0]*3//4,
                    image.shape[1]//4 : image.shape[1]*3//4]
sigma_ruido = np.std(centro) * 0.1
sigma_ruido = max(sigma_ruido, 0.02)

# Corrección de sesgo (mejora separación tumor/cerebro)
squared   = image**4 - 2 * (sigma_ruido**2)
corrected = np.sqrt(np.maximum(squared, 0))

# NL-means
h_nlm    = 0.8 * sigma_ruido
frame_nl = denoise_nl_means(
    corrected, h=h_nlm, sigma=sigma_ruido,
    fast_mode=True, patch_size=5, patch_distance=11, channel_axis=None
)

# Restaurar rango dinámico original
frame_nl = (frame_nl * (f_max - f_min)) + f_min

# Normalizar a [0,1] nuevamente para el resto del pipeline
frame_gauss = (frame_nl - frame_nl.min()) / (frame_nl.max() - frame_nl.min() + 1e-9)

print("  Preprocesamiento completado ✓")
print(f"  sigma_ruido: {sigma_ruido:.6f}")
print(f"  h_nlm: {h_nlm:.6f}")
# ══════════════════════════════════════════════════════════════
#  PASO 3 — SKULL STRIPPING + SEGMENTACIÓN  (de Nato)
# ══════════════════════════════════════════════════════════════
print("\n[2/4] Skull stripping...")

# Umbral de Otsu sobre el frame preprocesado
umbral_otsu    = filters.threshold_otsu(frame_gauss)
imagen_binaria = frame_gauss > umbral_otsu

# Morfología: erosión + closing
img_erosion = erosion(imagen_binaria, footprint_rectangle((1, 2)))
img_closing = closing(img_erosion,    footprint_rectangle((1, 2)))

# Etiquetado por solidez (método de Nato)
etiquetas    = label(img_closing, connectivity=2)
propiedades  = regionprops(etiquetas)
solidez      = np.array([r.solidity for r in propiedades])

if len(solidez) > 2:
    # Eliminar el más sólido (cráneo exterior) y el menos sólido (ruido)
    eliminar = (solidez != solidez.max()) & (solidez != solidez.min())
    solidez_filtrada = solidez * eliminar
else:
    solidez_filtrada = solidez

mascara = np.zeros_like(img_closing, dtype=bool)
for i in range(len(solidez_filtrada)):
    if solidez_filtrada[i] > 0:
        mascara |= (etiquetas == (i + 1))

mascara_rellena = binary_fill_holes(mascara)
frame_stripped  = frame_gauss * mascara_rellena.astype(float)

print("  Skull stripping completado ✓")

# ── Detección del tumor: patrón de anillo ──────────────────
print("\n[3/4] Detectando tumor (patrón anillo)...")

brain_vals  = frame_stripped[mascara_rellena]
thresh_high = np.percentile(brain_vals, 22) # mas alto valor , más exigente → solo lo más brillante, mayor a 50 no detecta nada xq la region tumoral no es tan brillante,  50 es el limite maximo
thresh_low  = np.percentile(brain_vals, 15)

# Borde brillante del tumor
bright_mask = (frame_stripped > thresh_high) & mascara_rellena

# Centro oscuro rodeado de zona brillante (necrosis)
from scipy.ndimage import binary_dilation
bright_dilated = binary_dilation(bright_mask, morphology.disk(5))# radio más chico
dark_core      = (frame_stripped < thresh_low) & bright_dilated & mascara_rellena & ~bright_mask

# Unir borde + núcleo
tumor_candidate = bright_mask | dark_core

# Limpieza morfológica
tumor_candidate = opening(tumor_candidate, morphology.disk(1))
tumor_candidate = binary_fill_holes(tumor_candidate)
tumor_candidate = closing(tumor_candidate, morphology.disk(3))

# Filtro por tamaño mínimo
labeled_tumor, n_reg = label(tumor_candidate, connectivity=2), None
labeled_tumor = label(tumor_candidate, connectivity=2)
n_reg         = labeled_tumor.max()

tumor_mask = np.zeros_like(tumor_candidate, dtype=bool)
min_voxels = 80

h, w = frame_stripped.shape
for reg in regionprops(labeled_tumor):
    if reg.area < min_voxels:
        continue

    # Filtro 1 — excentricidad: descarta líneas y estructuras alargadas
    if reg.eccentricity > 0.90:
        continue

    # Filtro 2 — varianza interna alta: el tumor tiene centro oscuro + borde brillante
    region_vals = frame_stripped[labeled_tumor == reg.label]
    if region_vals.std() < 0.04:
        continue

    # Filtro 3 — no estar en el centro sagital (hoz cerebral)
    cx_r = reg.centroid[0]
    fraccion_x = cx_r / h
    if 0.40 < fraccion_x < 0.60:
        continue

    # Filtro 4 — intensidad media suficientemente alta
    if region_vals.mean() < thresh_high * 0.80:
        continue

    tumor_mask[labeled_tumor == reg.label] = True

print(f"  Regiones detectadas: {label(tumor_mask).max()} ✓")


# ══════════════════════════════════════════════════════════════
#  PASO 4 — LOCALIZACIÓN ANATÓMICA  (de Santino)
# ══════════════════════════════════════════════════════════════
print("\n[4/4] Localizando regiones anatómicamente...")

labeled_final = label(tumor_mask, connectivity=2)
regiones_info = []

for reg in regionprops(labeled_final):
    cy_r, cx_r = reg.centroid   # regionprops devuelve (row, col) = (y, x)
    nombre, critica = localizar((cy_r, cx_r), affine)
    regiones_info.append({
        "id":       reg.label,
        "area":     reg.area,
        "centroid": (cx_r, cy_r),
        "nombre":   nombre,
        "critica":  critica,
    })

# Ordenar por área descendente
regiones_info.sort(key=lambda r: r["area"], reverse=True)

# Imprimir reporte
print(f"\n{'═'*55}")
print(f"  REPORTE — frame {FRAME}")
print(f"{'═'*55}")
if not regiones_info:
    print("  ✅ No se detectaron regiones anómalas.")
else:
    for i, reg in enumerate(regiones_info, 1):
        alerta = "⚠️  ZONA CRÍTICA" if reg["critica"] else "  "
        print(f"  Región {i}: {reg['area']} px  |  {reg['nombre']}  {alerta}")


# ══════════════════════════════════════════════════════════════
#  VISUALIZACIÓN FINAL  (overlay rojo de Santino)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='#1a1a2e')
fig.suptitle(f"Análisis frame {FRAME}  |  sub-KA02_run-02_T1w",
             color='white', fontsize=13, fontweight='bold')

titulos = ["Original", "Preprocesado (NL + Gauss)", "Segmentación + Localización"]
# Normalizar el original solo para mostrarlo
frame_display = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)

imagenes = [frame_display, frame_gauss, frame_gauss]

for ax, img_data, titulo in zip(axes, imagenes, titulos):
    ax.imshow(img_data.T, cmap='gray', origin='lower', aspect='auto')
    ax.set_title(titulo, color='white', fontsize=10)
    ax.axis('off')
    ax.set_facecolor('#0d0d1a')

# Overlay rojo en el tercer panel
overlay = np.zeros((*tumor_mask.shape, 4))
overlay[tumor_mask] = [1.0, 0.15, 0.15, 0.65]   # rojo semitransparente
axes[2].imshow(overlay.transpose(1, 0, 2), origin='lower', aspect='auto')

# Etiquetas sobre cada región
for reg in regiones_info:
    cx_r, cy_r = reg["centroid"]
    color_txt  = '#ff4444' if reg["critica"] else '#ffcc00'
    nombre_corto = reg["nombre"][:25]
    axes[2].annotate(
        f"R{reg['id']} — {nombre_corto}",
        xy=(cy_r, cx_r),
        xytext=(cy_r + 15, cx_r - 15),
        color=color_txt, fontsize=7,
        arrowprops=dict(arrowstyle='->', color=color_txt, lw=1.0),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.7),
        clip_on=True
    )

# Leyenda
patch_tumor = mpatches.Patch(color=(1, 0.15, 0.15, 0.65), label='Región sospechosa')
axes[2].legend(handles=[patch_tumor], loc='lower left',
               facecolor='#1a1a2e', labelcolor='white', fontsize=8)

# Panel de texto con reporte
if regiones_info:
    reporte = ""
    for i, reg in enumerate(regiones_info, 1):
        alerta = " ⚠ CRÍTICA" if reg["critica"] else ""
        reporte += f"R{i}: {reg['nombre']}{alerta}\n     {reg['area']} px\n\n"
    fig.text(0.01, 0.15, reporte.strip(),
             color='white', fontsize=7.5, va='top',
             bbox=dict(facecolor='#0d0d1a', alpha=0.8, pad=5))

plt.tight_layout()
plt.show()