import cv2
from medpy.filter.smoothing import anisotropic_diffusion
import numpy as np
from skimage import filters
from skimage.morphology import closing, footprint_rectangle, erosion
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import binary_fill_holes
from nilearn import datasets
from nilearn.image import load_img
import nibabel as nib

#~~~~~~~~~~~~~~~~~~~~~~
#   PREPROCESAMIENTO
#~~~~~~~~~~~~~~~~~~~~~~

def preprocesar(frame):
    difusion = anisotropic_diffusion(frame, niter=10, kappa=30, gamma=0.1)
    #anisotropic_diffusion devuelve 64bits, normalizar por las dudas
    return cv2.normalize(difusion, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#~~~~~~~~~~~~~~~~~~
#   SEGMENTACIÓN
#~~~~~~~~~~~~~~~~~~

def segmentar_craneo(corte):
#umbral
    umbral = filters.threshold_otsu(corte)
    imagen_binaria = corte > umbral
    
#aplicamos morfologia
    imagen_binaria_erosionada = erosion(imagen_binaria, footprint_rectangle((1,2)))
    imagen_binaria_closing = closing(imagen_binaria_erosionada, footprint_rectangle((1,2)))

#etiquetado y mascara
    etiquetas = label(imagen_binaria_closing, connectivity=2)
    propiedades = regionprops(etiquetas)

    solidez = np.array([region.solidity for region in propiedades])
    eliminar = (solidez != solidez.max()) & (solidez != solidez.min())
    solidez = solidez*eliminar

    mascara = 0
    n = 0
    for i in range(len(solidez)):
        if solidez[i] != solidez.min():
            n = etiquetas == (i+1)
        mascara = mascara + n
    
    mascara_rellena = binary_fill_holes(mascara)
    resultado = corte * mascara_rellena
    
    return resultado

def aumentar_contraste(frame):
    min, max = frame.min(), frame.max()
    frame = (frame - min)/(max - min)

    '''estimar el ruido sigma en el fondo: hay que asegurarse de que la esquina corresponda a una zona sin anatomía'''
    esquina = frame[0:40, 0:40]
    mean_esquina = np.mean(esquina)
    sigma = mean_esquina / np.sqrt(np.pi/2)

    '''correción de sesgo: al modificar frame** X, mejoramos la "separacion" entre tumor y cerebro'''
    squared = frame**4 - 2*(sigma**2) 
        
    cerebro_contraste_aumentado = np.sqrt(np.maximum(squared,0))
    return cerebro_contraste_aumentado

def segmentacion_tumor(cerebro_contraste_aumentado): 
    umbrales = filters.threshold_multiotsu(cerebro_contraste_aumentado, classes=3)
    imagen_binaria = cerebro_contraste_aumentado > umbrales[1]
    etiquetas = label(imagen_binaria, connectivity=2)
    propiedades = regionprops(etiquetas)
    areas = np.array([region.area for region in propiedades])
    indice_area_mayor = np.argmax(areas) + 1
    mascara = etiquetas == indice_area_mayor
    mascara_tumor = binary_fill_holes(mascara)
    return mascara_tumor

#~~~~~~~~~~~~~~~~~~~~~~~~~~
#   ANÁLISIS DE TEXTURAS
#~~~~~~~~~~~~~~~~~~~~~~~~~~
def analisis_texturas(frame,mascara_tumor): 
    resultado_glcm = []
    features = ["contrast",
                "homogeneity",
                "energy"]
    
    mascara_region_sana = np.fliplr(mascara_tumor)
    tumor_segmentado = frame*mascara_tumor
    region_sana_segmentada = frame*mascara_region_sana

    tumor_segmentado = ((tumor_segmentado/np.max(frame))*15).astype(np.uint8)
    region_sana_segmentada = ((region_sana_segmentada/np.max(frame))*15).astype(np.uint8)

    props_tumor = regionprops(mascara_tumor.astype(int))[0]
    min_row, min_col, max_row, max_col = props_tumor.bbox
    tumor_crop = tumor_segmentado[min_row:max_row, min_col:max_col]
    props_sano = regionprops(mascara_region_sana.astype(int))[0]
    min_row, min_col, max_row, max_col = props_sano.bbox
    sano_crop = region_sana_segmentada[min_row:max_row, min_col:max_col]


    glcm_tumor = graycomatrix(tumor_crop, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 16, symmetric = True, normed = True)
    glcm_sano =  graycomatrix(sano_crop, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 16, symmetric = True, normed = True)

    for feature in features:
        featuer_tumor = graycoprops(glcm_tumor, feature)
        feature_sano = graycoprops(glcm_sano, feature)
        promedio_tumor = featuer_tumor.mean()
        promedio_sano = feature_sano.mean()
        resultado_glcm.append([feature, promedio_sano, promedio_tumor])

    return resultado_glcm, props_tumor

def detectar_tumor(corte, mascara_tumor, resultado_glcm):
    tumor_detectado = 0
    resultado_glcm = analisis_texturas(corte,mascara_tumor)

    homogeneidad = resultado_glcm[1][2]
    print(homogeneidad)
    if homogeneidad > 0.68:
        tumor_detectado = 0 #no se detecto tumor
    else:
        tumor_detectado = 1 #se detecto tumor

    return tumor_detectado

#~~~~~~~~~~~
#   ATLAS
#~~~~~~~~~~~

def atlas(fila_tumor, columna_tumor, numero_de_corte, file_path):
    atlas_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm') #subcortical

    atlas_sub_img   = load_img(atlas_sub.maps)
    data = nib.load(file_path)

    paciente_affine = data.affine
    atlas_affine = atlas_sub_img.affine
    atlas_affine_inv = np.linalg.inv(atlas_affine)
    atlas_sub_data  = atlas_sub_img.get_fdata()
    labels = atlas_sub.labels
    voxel_tumor = np.array([fila_tumor, columna_tumor, numero_de_corte,1])
    voxel_tumor_MNI = paciente_affine @ voxel_tumor
    voxel_atlas = atlas_affine_inv @ voxel_tumor_MNI #esto tiene las coordenadas del atlas para nuestra region

    ax = int(voxel_atlas[0])
    ay = int(voxel_atlas[1])
    az = int(voxel_atlas[2])

    numero_region = int(atlas_sub_data[ax, ay, az])
    nombre_region = labels[numero_region]
    print(nombre_region)

    return nombre_region

#~~~~~~~~~~~~~~~
#   UNIFICADO
#~~~~~~~~~~~~~~~

def total(corte):
    prep = preprocesar(corte)
    sin_craneo = segmentar_craneo(prep)
    contraste = aumentar_contraste(sin_craneo)
    tumor = segmentacion_tumor(contraste)

    alpha = 0.4 #transparencia
    corte_norm = (corte - corte.min()) / (corte.max() - corte.min())
    imagen_rgb = np.stack([corte_norm] * 3, axis=-1)

    overlay = imagen_rgb.copy()
    overlay[tumor == 1] = [1.0, 0.0, 0.0]
    
    resultado = (1 - alpha) * imagen_rgb + alpha * overlay
    
    return resultado, tumor

