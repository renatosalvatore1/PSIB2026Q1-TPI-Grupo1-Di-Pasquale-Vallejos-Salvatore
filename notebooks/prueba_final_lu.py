
import numpy as np
from skimage import filters
from skimage.morphology import closing, footprint_rectangle, erosion
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes


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


