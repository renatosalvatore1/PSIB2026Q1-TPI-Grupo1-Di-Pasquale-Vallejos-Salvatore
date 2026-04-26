import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

data = nib.load("./data/sub-KA72/anat/sub-KA72_run-02_T1w.nii.gz").get_fdata()

fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.15)

corte_inicial = data.shape[2] // 2
img_plot = ax.imshow(data[:, :, corte_inicial], cmap='gray')
ax.set_title(f"Corte {corte_inicial}")
ax.axis('off')

# Crear slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Corte', 0, data.shape[2] - 1,
                valinit=corte_inicial, valstep=1)

def actualizar(val):
    c = int(slider.val)
    img_plot.set_data(data[:, :, c])
    ax.set_title(f"Corte {c}")
    fig.canvas.draw_idle()

slider.on_changed(actualizar)
plt.show()