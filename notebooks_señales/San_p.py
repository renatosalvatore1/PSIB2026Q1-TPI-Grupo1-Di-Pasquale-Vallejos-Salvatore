'''
Hello, Here are the details of the positions of the external electrodes:
Ext1 --> left eye corner
Ext 2 --> right eye corner
Ext 3 --> left eye eyebrow (above)
Ext 4 --> left eye below
Ext 7 --> middle of the collar bone
Ext 5 --> left mastoide
Ext 6 --> right mastoide
EXT 8 was set up with an extra electrode for Fp1, due to a defect of that electrode on the Biosemi electrode set.
So from that you have:
 Ext1 and Ext2: HEOG; 
 Ext3 and Ext 4: VEOG (blinks); 
Ext 7: ECG
Ext5: M1
Ext6: M2
Ext8: FP1

'''
'''
Sujetos de 25-55 no tienen experiencia de meditacion, todo el resto tienen, pero distintos años de experiencia, desde 1 año hasta 56 años.
'''

'''
cada archivo sub es 1 sujeto
(which makes up a total of 100 subjects)
'''

import mne

# cargar el archivo .bdf
raw = mne.io.read_raw_bdf('./ds003969/sub-001/eeg/sub-001_task-med1breath_eeg.bdf', preload=True)

# recortar el primer minuto
raw_min1 = raw.crop(tmin=0, tmax=10)


# info básica del dataset
print(raw.info)

# graficar los canales en el tiempo (interactivo, podés scrollear)
raw_min1.plot(n_channels=30, duration=1, title='EEG - meditación respiración')


# graficar el espectro de frecuencias
psd = raw.compute_psd(fmin=0, fmax=50)
fig = psd.plot(show=False)

# resaltar la banda alfa
ax = fig.axes[0]

# limitar el eje Y (valores en dB)
ax.set_ylim(-20, 60)  # ajustá estos valores a lo que ves
ax.axvspan(8, 12, alpha=0.2, color='blue', label='alfa')
ax.legend()


import matplotlib.pyplot as plt
plt.show()