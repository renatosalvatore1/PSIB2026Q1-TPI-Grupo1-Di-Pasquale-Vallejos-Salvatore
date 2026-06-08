import matplotlib.pyplot as plt
import nibabel as nib
import cv2

from skimage.restoration import denoise_bilateral
from medpy.filter.smoothing import anisotropic_diffusion

def al_presionar_tecla(event):
    plt.close()

def normalizar(imagen):
    return cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def histograma(imagen):
    return cv2.calcHist([imagen],[0],None,[256],[0,256])

def show(imagen, histograma, titulo):
    fig,axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(imagen, cmap='grey')
    axs[0].axis('off')
    axs[1].plot(histograma,color='black')
    axs[1].set_xlim([7,256])
    axs[1].set_ylim([0,600])
    plt.title(titulo)
    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', al_presionar_tecla)
    plt.show()

def comparar(imagen1, imagen2, titulo):
    fig,axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(imagen1, cmap='grey')
    axs[0].axis('off')
    axs[1].imshow(imagen2, cmap='grey')
    axs[1].axis('off')
    plt.title(titulo)
    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', al_presionar_tecla)
    plt.show()

data = nib.load('./data/sub-KA02/anat/sub-KA02_run-02_T1w.nii.gz').get_fdata()
frame = data[:,:,200]

frame_n = normalizar(frame)
#h_og = histograma(frame_n)
#show(frame_n,h_og,"Frame 200, Imagen Original")

#   --- Bilateral filter ---
    #Calcula el promedio de los píxeles en función de su proximidad espacial y similitud radiométrica.
    #Conserva los bordes y reduce el ruido → ideal para MRI o ecografía.
bilateral = denoise_bilateral(frame_n, sigma_color=0.05, sigma_spatial=15)
        #sigma_color:   float
                        #Standard deviation for grayvalue/color distance (radiometric similarity).
                        #A larger value results in averaging of pixels with larger radiometric differences.
                        #If None, the standard deviation of image will be used.
        #sigma_spatial: float
                        #Standard deviation for range distance. 
                        #A larger value results in averaging of pixels with larger spatial differences.
        #devuelve float64, ver para qué lo usas y NORMALIZÁ

bilateral_n = normalizar(bilateral)
h_bilateral = histograma(bilateral_n)
#show(bilateral_n, h_bilateral, "Bilateral Filter")

#uno = denoise_bilateral(frame_n, sigma_color=0.05, sigma_spatial=2)
#comparar(bilateral_n, normalizar(uno), "sigma_spatial 15 vs sigma_spatial 2")


#   --- Filtro de Difusión Anisotrópica ---
    #Difunde la imagen suavemente dentro de regiones, pero no cruza bordes.
    #Transformación lineal e invariante espacial de la imagen original.
    #Combina ventajas de suavizado y preservación de bordes.
difusion = anisotropic_diffusion(frame_n, niter=15, kappa=30, gamma=0.1)
        #niter:     Number of iterations.
                    #A higher value results in a smoother image, but it may also introduce noise. 
        #kappa:     The higher the more edges are smoothed over
        #gamma:     The higher, the stronger the plateaus between edges are smeared.
        #devuelve float64, ver para qué lo usas y NORMALIZÁ


difusion_n = normalizar(difusion)
h_difusion = histograma(difusion_n)

#show(difusion_n,h_difusion,"Difusión Anisotrópica")

dos =  anisotropic_diffusion(frame_n, niter=10, kappa=30, gamma=0.05)
comparar(frame,dos,"Plateaus between edges are smeared stronger")