import cv2
import matplotlib.pyplot as plt
import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion

def preprocesar(frame):
    # Aplicar difusión anisotrópica
    difusion = anisotropic_diffusion(frame, niter=10, kappa=30, gamma=0.05)
    
    # Normalizar a 8 bits (valores de 0 a 255)
    frame_procesado = cv2.normalize(difusion, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 1. Calcular el histograma (256 bins para el rango 0-255)
    hist = cv2.calcHist([frame_procesado], [0], None, [256], [0, 256])
    
    # 2. Crear el rango de intensidades (el eje X, de 0 a 255)
    intensidades = np.arange(256)
    
    # 3. Filtrar para mostrar solo desde la intensidad 8 en adelante
    intensidades_filtradas = intensidades[8:]
    hist_filtrado = hist[8:].flatten() # .flatten() lo pasa a un array de 1D
    
    # 4. Graficar el histograma con barras
    plt.figure(figsize=(10, 5))
    plt.bar(intensidades_filtradas, hist_filtrado, width=1.0, color='gray', edgecolor='black')
    
    # Configuraciones de la ventana del gráfico
    plt.title("Histograma de la señal (Intensidad >= 8)")
    plt.xlabel("Intensidad de píxel")
    plt.ylabel("Cantidad de píxeles")
    plt.xlim(8, 255) # Forzamos a que el eje X arranque en 8
    plt.grid(axis='y', alpha=0.75)
    
    # Mostrar el gráfico
    plt.show()
    
    return frame_procesado