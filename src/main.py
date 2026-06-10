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
import sys
import nibabel as nib
import numpy as np
from PySide6 import QtWidgets
from interfaz import MyWidget
from operaciones import aumentar_contraste, segmentacion_tumor, segmentar_craneo, analisis_texturas, imagen_final
from atlas import atlas


def procesar(corte, indice, file_path):
    cerebro_segmentado = segmentar_craneo(corte)
    cerebro_contraste = aumentar_contraste(cerebro_segmentado)
    matriz_glcm,tumor_detectado,mascara_tumor,fila_tumor, columna_tumor = segmentacion_tumor(cerebro_contraste,corte, cerebro_segmentado)
    region_cerebro = atlas(fila_tumor, columna_tumor,indice,file_path,tumor_detectado)
    imagen_rgb = imagen_final(corte, mascara_tumor,tumor_detectado)
    return imagen_rgb, region_cerebro

app = QtWidgets.QApplication([])
widget = MyWidget(proceso=procesar)
widget.resize(900, 600)
widget.show()
app.exec()
