import nibabel as nib
import skimage 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, color, io
from skimage import segmentation
from skimage import morphology
from skimage.morphology import opening, closing, disk, footprint_rectangle, erosion
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from operaciones import aumentar_contraste,segmentacion_tumor,analisis_texturas, segmentar_craneo
from atlas import atlas

data = nib.load("./data/sub-KA02/anat/sub-KA02_run-02_T1w.nii.gz").get_fdata()
numero_de_corte = 200
corte = data[:,:,numero_de_corte]

'''ACA PONER EL CODIGO DE SEGMENTACION CRANEO'''
cerebro_segmentado = segmentar_craneo(corte)

#AUMENTAMOS CONTRASTE
cerebro_contraste = aumentar_contraste(cerebro_segmentado)
#SEGMENTAMOS EL TUMOR #ANALIZAMOS LA TEXTURA DEL TUMOR PARA EXTRAER CARACTERISTICAS Y VALIDAR EL RESULTADO OBTENIDO
matriz_glcm,tumor_detectado = segmentacion_tumor(cerebro_contraste,corte, cerebro_segmentado)
print(tumor_detectado)



