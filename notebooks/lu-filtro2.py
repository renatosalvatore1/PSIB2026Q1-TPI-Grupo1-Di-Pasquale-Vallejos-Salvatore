import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float

#analogo al WaitKey en cv2
def al_presionar_tecla(event):
    plt.close()

#CARGA DEL FRAME DESDE DICOM Y NORMALIZACIÓN_________________________________________________________
data = nib.load('./data/sub-KA02/anat/sub-KA02_run-02_T1w.nii.gz').get_fdata()
frame = data[:,:,200]
frame_n = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#FUNCIÓN NL-MEANS____________________________________________________________________________________
def nl_means(image,exponente):
    '''normalizar en el rango [0,1] para skimage, en vez de [0,255] o unidades Hounsfield'''
    min, max = image.min(), image.max()
    image = (image - min)/(max - min)

    '''estimar el ruido sigma en el fondo: hay que asegurarse de que la esquina corresponda a una zona sin anatomía'''
    esquina = image[0:40, 0:40]
    mean_esquina = np.mean(esquina)
    sigma = mean_esquina / np.sqrt(np.pi/2)

    '''correción de sesgo'''
    squared = image**exponente - 2*(sigma**2) 
    '''Al modificar image** X, mejoramos la "separacion" entre tumor y cerebro'''
    
    corrected = np.sqrt(np.maximum(squared,0))

    h = 0.8 * sigma
    
    filtrada = denoise_nl_means(corrected,h=h,sigma=sigma,fast_mode=True,patch_size=5,patch_distance=11,channel_axis=None)
    '''
    patch_size=5, patch_distance=11 estandar para mri estructural T1,T2.
    subir si la resolucion es muy alta o ruido es extremadamente grueso a 7 y 15 respectivamente, a costo de aumento de tiempo de cómputo
    '''    
    
    final = (filtrada * (max - min)) + min
    '''restaurar rango dinámico original si es necesario'''
    return final
    

filtro2=nl_means(frame_n,4)

fig,axs = plt.subplots(1,2,figsize=(15,10))
axs[0].imshow(frame_n, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(filtro2, cmap='gray')
axs[1].set_title('NL_means **4')

plt.tight_layout()
plt.show()