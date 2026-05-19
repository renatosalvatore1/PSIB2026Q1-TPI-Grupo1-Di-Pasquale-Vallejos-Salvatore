import nibabel as nib
import matplotlib.pyplot as plt
import cv2

#analogo al WaitKey en cv2
def al_presionar_tecla(event):
    plt.close()

data = nib.load('./data/sub-KA02/anat/sub-KA02_run-02_T1w.nii.gz').get_fdata()
frame = data[:,:,89]
frame_n = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #dicom trabaja con 16bits, cv2 con 8
hist = cv2.calcHist([frame_n],[0],None,[256],[0,256])

fig,axs=plt.subplots(1,2, figsize=(10,5))
axs[0].imshow(frame, cmap='grey')
axs[0].set_title('Frame - Original')
axs[0].axis('off')
axs[1].plot(hist,color='black')
axs[1].set_title('Histograma')
axs[1].set_xlim([0,256])
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

mediana = cv2.medianBlur(frame_n,ksize=5)
gaussiano = cv2.GaussianBlur(frame_n, ksize=(5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_CONSTANT)
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