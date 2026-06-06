import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float
'''
fun fact: nibabel suele cargar volúmenes tridimensionales, .get_data devuelve un arreglo 3D indexado
podríamos también cargar el dicom con la libreria pydicom, que lee los archivos corte por corte (2D)
'''

#analogo al WaitKey en cv2
def al_presionar_tecla(event):
    plt.close()

#_____________________________________________________________________________
#CARGA DEL FRAME DESDE DICOM Y NORMALIZACIÓN
#_____________________________________________________________________________

data = nib.load('./data/sub-KA02/anat/sub-KA02_run-02_T1w.nii.gz').get_fdata()
frame = data[:,:,200]
frame_n = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #dicom trabaja con 16bits, cv2 con 8

#_____________________________________________________________________________
#ANÁLISIS EN BASE AL HISTOGRAMA
#_____________________________________________________________________________
hist = cv2.calcHist([frame_n],[0],None,[256],[0,256])

fig,axs=plt.subplots(1,2, figsize=(10,5))
axs[0].imshow(frame, cmap='grey')
axs[0].set_title('Frame - Original')
axs[0].axis('off')
axs[1].plot(hist,color='black')
axs[1].set_title('Histograma')
axs[1].set_xlim([7,256])
axs[1].set_ylim([0,500])
plt.title('Frame 89')
plt.tight_layout()
fig.canvas.mpl_connect('key_press_event', al_presionar_tecla) #waitKey
plt.show()

'''
En base al histograma se pueden observar características correspondientes a ruido Gaussiano (distribución 
en forma de campana superpuesta a las intensidades dominantes de la imagen) y ruido Speckle, que afecta a
las zonas claras (cola larga hacia zonas de mayor intensidad y agrupamiento denso en las zonas de baja).
Para tratar estos ruidos se propone aplicar un filtro Gaussiano y un filtro mediana, respectivamente.
'''

#_____________________________________________________________________________
#APLICACIÓN DE FILTROS MEDIANA Y GAUSSIANO
#_____________________________________________________________________________
mediana = cv2.medianBlur(frame_n,ksize=3)

'''Esto modificar para que no se borrene tanto, sigma y alpha'''
gaussiano = cv2.GaussianBlur(frame_n, ksize=(3, 3), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_CONSTANT)

hist_m = cv2.calcHist([mediana],[0],None,[256],[0,256])
hist_g = cv2.calcHist([gaussiano],[0],None,[256],[0,256])

fig,axs=plt.subplots(3,2, figsize=(15,10))
axs[0][0].imshow(frame, cmap='grey')
axs[0][0].axis('off')
axs[0][0].set_title('Original')
axs[0][1].plot(hist,color='black')
axs[0][1].set_xlim([0,256])

axs[1][0].imshow(mediana, cmap='grey')
axs[1][0].axis('off')
axs[1][0].set_title('Filtrado con mediana')
axs[1][1].plot(hist_m,color='black')
axs[1][1].set_xlim([0,256])

axs[2][0].imshow(gaussiano, cmap='grey')
axs[2][0].axis('off')
axs[2][0].set_title('Filtrado con gaussiano')
axs[2][1].plot(hist_g,color='black')
axs[2][1].set_xlim([0,256])

plt.tight_layout()
fig.canvas.mpl_connect('key_press_event', al_presionar_tecla) #waitKey
plt.show()


final = cv2.medianBlur(frame_n,ksize=5)
hist_f = cv2.calcHist([final],[0],None,[256],[0,256])

fig,axs=plt.subplots(1,3, figsize=(15,10))
axs[0].imshow(frame, cmap='grey')
axs[0].set_title('Frame - Original')
axs[0].axis('off')
axs[1].imshow(final, cmap='grey')
axs[1].set_title('Filtrado')
axs[1].axis('off')
axs[2].plot(hist_f,color='black')
axs[2].set_title('Histograma')
axs[2].set_xlim([0,256])
plt.title('Frame 89 + Filtrado')
plt.tight_layout()
fig.canvas.mpl_connect('key_press_event', al_presionar_tecla) #waitKey
plt.show()


#_____________________________________________________________________________
#APLICACIÓN DE NL-MEANS
#_____________________________________________________________________________
#normalizar en el rango [0,1] para skimage, en vez de [0,255] o unidades Hounsfield
min, max = frame.min(), frame.max()
image = (frame - min)/(max - min)

#estimar el ruido sigma en el fondo: hay que asegurarse de que la esquina corresponda a una zona sin anatomía
esquina = image[0:40, 0:40]
mean_esquina = np.mean(esquina)
sigma = mean_esquina / np.sqrt(np.pi/2)

#correción de sesgo
squared = image**1 - 2*(sigma**2) 
'''
Al modificar image** X, mejoramos la "separacion" entre tumor y cerebro
'''
corrected = np.sqrt(np.maximum(squared,0))

h = 0.8 * sigma #filtrado Non-Local Means

filtrada = denoise_nl_means(corrected,h=h,sigma=sigma,fast_mode=True,patch_size=5,patch_distance=11,channel_axis=None)
    #patch_size=5, patch_distance=11 estandar para mri estructural T1,T2.
    #subir si la resolucion es muy alta o ruido es extremadamente grueso a 7 y 15 respectivamente
    #a costo de aumento de tiempo de cómputo

final = (filtrada * (max - min)) + min#restaurar rango dinámico original si es necesario

fig,axs = plt.subplots(1,2,figsize=(18,6))
axs[0].imshow(frame, cmap='gray')
axs[0].set_title('Original')
axs[0].axis('off')

axs[1].imshow(final, cmap='gray')
axs[1].set_title('Filtrado con NL-means')
axs[1].axis('off')

plt.tight_layout()
plt.show()