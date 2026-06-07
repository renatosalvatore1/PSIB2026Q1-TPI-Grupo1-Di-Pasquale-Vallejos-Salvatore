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
RUTA_ARCHIVO = "./data/sub-KA33/anat/sub-KA33_run-02_T1w.nii.gz"
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


'''
El atlas es un volumen 3D donde cada voxel tiene un numero entero que indica a que region antomica pertenece
_atlas_data[x, y, z] = 23   → "Precentral Gyrus"
_atlas_data[x, y, z] = 0    → fuera del cerebro / no especificado
_atlas_data[x, y, z] = 47   → "Hippocampus"

AAL_LABELS es el diccionario que traduce ese numero a un nombre
AAL_LABELS = {
    0:  "",
    1:  "Frontal Pole",
    2:  "Insular Cortex",
    23: "Precentral Gyrus",
    47: "Hippocampus",
    ...
}

'''

def is_critical(label_name: str) -> bool:
    return any(kw.lower() in label_name.lower() for kw in CRITICAL_KEYWORDS)

def localizar(centroid_vox, affine):
    cy, cx = centroid_vox
    sx, sy, sz = data.shape[:3]
    ax_dim, ay_dim, az_dim = _atlas_data.shape[:3]

    # Proporción directa vóxel paciente → vóxel atlas
    # El atlas Harvard-Oxford cort está en espacio MNI 2mm, 91x109x91
    ax = int(round(cx   / sx * ax_dim))
    ay = int(round(cy   / sy * ay_dim))
    az = int(round(FRAME / sz * az_dim))

    print(f"  Vóxel paciente ({cx:.0f}, {cy:.0f}, {FRAME}) → atlas ({ax}, {ay}, {az})")

    if not (0 <= ax < ax_dim and 0 <= ay < ay_dim and 0 <= az < az_dim):
        return "Fuera del espacio del atlas", False

    label_idx = int(_atlas_data[ax, ay, az])
    if label_idx == 0:
        for dx, dy, dz in [(5,0,0),(-5,0,0),(0,5,0),(0,-5,0),(0,0,5),(0,0,-5)]:
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
squared   = image**2 - 2 * (sigma_ruido**2)
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
#  PASO 3 — DESPRENDIMIENTO DEL CRÁNEO + SEGMENTACIÓN  (de Nato)
# ══════════════════════════════════════════════════════════════
print("\n[2/4] DESPRENDIMIENTO DEL CRÁNEO...")

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

# DEBUG — ver distribución de valores
brain_vals = frame_stripped[mascara_rellena]
print(f"  brain_vals min: {brain_vals.min():.4f}")
print(f"  brain_vals max: {brain_vals.max():.4f}")
print(f"  brain_vals mean: {brain_vals.mean():.4f}")
print(f"  percentil 80: {np.percentile(brain_vals, 80):.4f}")
print(f"  percentil 85: {np.percentile(brain_vals, 85):.4f}")
print(f"  percentil 90: {np.percentile(brain_vals, 90):.4f}")
print(f"  percentil 95: {np.percentile(brain_vals, 95):.4f}")

# Ver valor en la zona del tumor visualmente (aproximado)
# El tumor se ve en la parte inferior izquierda del frame
# Tomamos un parche en esa zona
h, w = frame_stripped.shape
parche = frame_stripped[h//4:h//2, w//4:w//2]
print(f"  Max en parche inferior-izq: {parche.max():.4f}")
print(f"  Mean en parche inferior-izq: {parche.mean():.4f}")



print("  DESPRENDIMIENTO DEL CRÁNEO completado ✓")

# ── Detección del tumor: patrón de anillo ──────────────────
print("\n[3/4] Detectando tumor (patrón anillo)...")

brain_vals  = frame_stripped[mascara_rellena]
thresh_high = np.percentile(brain_vals, 95) # mas alto valor , más exigente → solo lo más brillante, mayor a 50 no detecta nada xq la region tumoral no es tan brillante,  50 es el limite maximo
thresh_low  = np.percentile(brain_vals, 10)

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

brain_rows = np.where(mascara_rellena.any(axis=1))[0]
brain_cols = np.where(mascara_rellena.any(axis=0))[0]
brain_row_min, brain_row_max = brain_rows[0], brain_rows[-1]
brain_height = brain_row_max - brain_row_min


h, w = frame_stripped.shape
for reg in regionprops(labeled_tumor):
    if reg.area < min_voxels:
        continue
    
    cy_r, cx_r = reg.centroid
    print(f"  brain_row_min: {brain_row_min}, brain_row_max: {brain_row_max}")
    print(f"  brain_height: {brain_height}")
    pos_relativa = (cy_r - brain_row_min) / brain_height
    print(f"  Región {reg.label}: cy_r={cy_r:.1f}, pos_relativa={pos_relativa:.2f}, area={reg.area}")
    if pos_relativa > 0.75:
        continue
    # Filtro posición — descartar 25% superior del bounding box (zona de ojos)
    '''
    cy_r=43.0, pos_relativa=0.19 → está en el 19% inferior del bounding box, no en el superior
    El bounding box va de fila 13 a 167, y el ojo está en fila 43 — que es la parte inferior de la imagen transpuesta

    El problema es que la imagen se muestra con .T (transpuesta) en matplotlib, pero regionprops trabaja sobre el array sin transponer. Entonces "arriba" en la imagen visual es "abajo" en las coordenadas del array.
    El ojo está en pos_relativa=0.19 — o sea en el 19% desde brain_row_min. Hay que invertir el filtro, cambiando >0,75 a <0,25
    '''
    pos_relativa = (cy_r - brain_row_min) / brain_height
    if pos_relativa < 0.25:#descarta el 25% inferior del bounding box ( zona de los ojos en coord de array)
        continue

    # Filtro borde lateral — descartar regiones muy pegadas al borde izquierdo o derecho
    fraccion_lateral = cx_r / w
    if fraccion_lateral < 0.15 or fraccion_lateral > 0.85:
        continue

    # Filtro máscara — el centroide debe estar dentro del cerebro real
    cy_int, cx_int = int(round(cy_r)), int(round(cx_r))
    if not mascara_rellena[cy_int, cx_int]:
        continue

    # Filtro 1 — excentricidad: descarta líneas y estructuras alargadas
    if reg.eccentricity > 0.90:
        continue

    # Filtro 2 — varianza interna alta: el tumor tiene centro oscuro + borde brillante
    region_vals = frame_stripped[labeled_tumor == reg.label]
    if region_vals.std() < 0.04:
        continue

   # Filtro 3 — no estar en el centro sagital (hoz cerebral)
    cy_r, cx_r = reg.centroid    # cy_r = fila, cx_r = columna
    fraccion_x = cx_r / w        # dividir columna por ancho
    if 0.40 < fraccion_x < 0.60 and reg.area < 300:
        continue

    # Filtro 4 — intensidad media suficientemente alta
    if region_vals.mean() < thresh_high * 0.50:
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

# Overlay con dos colores: rojo=crítico, naranja=no crítico
overlay = np.zeros((*tumor_mask.shape, 4))

labeled_overlay = label(tumor_mask, connectivity=2)
for reg in regionprops(labeled_overlay):
    cy_r, cx_r = reg.centroid
    _, es_critica = localizar((cy_r, cx_r), affine)
    mascara_region = labeled_overlay == reg.label
    overlay[mascara_region] = [1.0, 0.10, 0.10, 0.70]  # rojo siempre

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
patch_tumor    = mpatches.Patch(color=(1.0, 0.10, 0.10, 0.70), label='Tumor detectado')
patch_critico  = mpatches.Patch(color=(1.0, 0.55, 0.00, 0.70), label='Zona crítica adyacente')
axes[2].legend(handles=[patch_tumor, patch_critico], loc='lower left',
               facecolor='#1a1a2e', labelcolor='white', fontsize=8)

# Panel de texto con reporte
if regiones_info:
    reporte = ""
    for i, reg in enumerate(regiones_info, 1):
        alerta = " ⚠️  CRÍTICA" if reg["critica"] else ""
        reporte += f"R{i}: {reg['nombre']}{alerta}\n     {reg['area']} px\n\n"
    fig.text(0.01, 0.15, reporte.strip(),
             color='white', fontsize=7.5, va='top',
             bbox=dict(facecolor='#0d0d1a', alpha=0.8, pad=5))

plt.tight_layout()
plt.show()