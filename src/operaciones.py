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
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
#COLSULTAR SI PODEMOS USAR ESTO
from scipy.ndimage import binary_fill_holes

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
#COLSULTAR SI PODEMOS USAR ESTO
from scipy.ndimage import binary_fill_holes





def segmentar_craneo(corte):
#umbral
    umbral = filters.threshold_otsu(corte)
    imagen_binaria = corte > umbral
    print("IMAGEN UMBRALIZADA")
    plt.imshow(imagen_binaria[:,:],cmap='gray')
    plt.show()

#aplicamos morfologia
    imagen_binaria_erosionada = erosion(imagen_binaria, footprint_rectangle((1,2)))
    print("POST EROSION")
    plt.imshow(imagen_binaria_erosionada[:,:],cmap='gray')
    plt.show()
    imagen_binaria_closing = closing(imagen_binaria_erosionada, footprint_rectangle((1,2)))
    print("POST CLOSING")
    plt.imshow(imagen_binaria_closing[:,:],cmap='gray')
    plt.show()
    
#etiquetado y mascara
    etiquetas = label(imagen_binaria_closing, connectivity=2)
    propiedades = regionprops(etiquetas)
    #areas = [region.area for region in propiedades]
    #print("Cantidad de componentes:", len(areas))
    #print("Area de la componente mayor:", max(areas))
    #indice = np.argmax(areas)
    solidez = np.array([region.solidity for region in propiedades])
    print(solidez)
    eliminar = (solidez != solidez.max()) & (solidez != solidez.min())
    print(eliminar)
    solidez = solidez*eliminar
    print(solidez)
    indice = np.argmax(solidez)
    mascara = 0
    n = 0
    for i in range(len(solidez)):
        if solidez[i] != solidez.min():
            n = etiquetas == (i+1)
        mascara = mascara + n
    print("MASCARA ANTES DE FILL HOLES")
    plt.imshow(mascara[:,:], cmap='gray')
    plt.show()
    mascara_rellena = binary_fill_holes(mascara)
    resultado = corte * mascara_rellena
    print("MASCARA")
    plt.imshow(mascara_rellena[:,:],cmap='gray')
    plt.show()
    
    resultado = corte * mascara_rellena
    print("RESULTADO")
    plt.imshow(resultado[:,:],cmap='gray')
    plt.show()
    print("corte")
    plt.imshow(corte[:,:],cmap='gray')
    plt.show()
    return resultado






#AUMENTAR CONTRASTE
def aumentar_contraste(frame):
    min, max = frame.min(), frame.max()
    frame = (frame - min)/(max - min)

    '''estimar el ruido sigma en el fondo: hay que asegurarse de que la esquina corresponda a una zona sin anatomía'''
    esquina = frame[0:40, 0:40]
    mean_esquina = np.mean(esquina)
    sigma = mean_esquina / np.sqrt(np.pi/2)

    '''correción de sesgo'''
    squared = frame**4 - 2*(sigma**2) 
    '''Al modificar frame** X, mejoramos la "separacion" entre tumor y cerebro'''
    
    cerebro_contraste_aumentado = np.sqrt(np.maximum(squared,0))
    return cerebro_contraste_aumentado


#ANALISIS DE TEXTURAS
resultado_glcm = []
def analisis_texturas(frame,mascara_tumor): 
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
        #desvio_tumor = featuer_tumor.std()
        promedio_sano = feature_sano.mean()
        #desvio_sano = feature_sano.std()
        resultado_glcm.append([feature, promedio_sano, promedio_tumor])

    return resultado_glcm

#SEGMENTACION TUMOR

def segmentacion_tumor(cerebro_contraste_aumentado,corte,cerebro_segmentado): 
    tumor_detectado = 0
    umbrales = filters.threshold_multiotsu(cerebro_contraste_aumentado, classes=3)
    imagen_binaria = cerebro_contraste_aumentado > umbrales[1]
    etiquetas = label(imagen_binaria, connectivity=2)
    propiedades = regionprops(etiquetas)
    areas = np.array([region.area for region in propiedades])
    indice_area_mayor = np.argmax(areas) + 1
    mascara = etiquetas == indice_area_mayor
    mascara_tumor = binary_fill_holes(mascara)
    tumor_segmentado = cerebro_segmentado * mascara_tumor

    solidez = np.array([region.solidity for region in propiedades])
    #print(solidez[indice_area_mayor-1])
    plt.imshow(mascara,cmap='gray')
    plt.show()
    plt.imshow(mascara_tumor,cmap='gray')
    plt.show()
    plt.imshow(tumor_segmentado,cmap='gray')
    plt.show()
    #llamamos glcm
    resultado_glcm = analisis_texturas(corte,mascara_tumor)

    homogeneidad = resultado_glcm[1][2]
    print(homogeneidad)
    if homogeneidad > 0.68:
        tumor_detectado = 0 #no se detecto tumor
    else:
        tumor_detectado = 1 #se detecto tumor

    return resultado_glcm,tumor_detectado





