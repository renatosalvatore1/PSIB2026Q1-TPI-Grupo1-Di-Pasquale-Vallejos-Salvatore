import cv2
from medpy.filter.smoothing import anisotropic_diffusion

def preprocesar(frame):
    difusion = anisotropic_diffusion(frame, niter=10, kappa=30, gamma=0.05)
    #anisotropic_diffusion devuelve 64bits, normalizar por las dudas
    return cv2.normalize(difusion, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)



