import sys
import os
from PySide6 import QtCore, QtWidgets, QtGui
import numpy as np
import nibabel as nib


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()


        #   ---Widgets---
        self.button = QtWidgets.QPushButton("Seleccionar archivo .nii")
        
        self.title = QtWidgets.QLabel("Interfaz Diagnóstico de Tumor Cerebral y Localización", alignment=QtCore.Qt.AlignCenter)
        
        self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: black;")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)

        self.slice_info = QtWidgets.QLabel("Frame: -", alignment=QtCore.Qt.AlignCenter)

        #   ---Layout---
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.slice_info)
        
        #   ---Conexiones---
        self.button.clicked.connect(self.select_file)
        self.slider.valueChanged.connect(self.update_slice)

        #   ---Guarda el array 3D del .nii---
        self.volume = None

    def select_file(self):
        starting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo .nii",
            starting_dir,
            "Todos los archivos (*)" #Filtro
        )

        if file_path:
            self.load_nii(file_path)
    
    def load_nii(self, file_path): #acá podríamos poner TODO nuestro análisis
        img = nib.load(file_path)
        data = img.get_fdata()

        if data.ndim == 4: # Soporte para volúmenes 3D (x, y, z) y 4D (x, y, z, t)
            data = data[:, :, :, 0] # Tomamos el primer volumen temporal y navegamos en z

        self.volume = data
        num_slices = data.shape[2]  # Eje axial (z)

        # Configurar el slider
        self.slider.setMinimum(0)
        self.slider.setMaximum(num_slices - 1)
        self.slider.setValue(num_slices // 2)  # Empieza en el slice del medio
        self.slider.setEnabled(True)

        self.update_slice(num_slices // 2)

    def update_slice(self, index):
        if self.volume is None:
            return

        slice_data = self.volume[:, :, index]

        # Normalizar a 0-255 para visualización
        slice_min, slice_max = slice_data.min(), slice_data.max()
        if slice_max > slice_min:
            slice_norm = (slice_data - slice_min) / (slice_max - slice_min) * 255
        else:
            slice_norm = np.zeros_like(slice_data)
        slice_norm = slice_norm.astype(np.uint8)

        # Rotar para orientación correcta (los .nii suelen venir transpuestos)
        slice_norm = np.rot90(slice_norm)

        # Convertir a QPixmap para mostrarlo en el QLabel
        height, width = slice_norm.shape
        q_image = QtGui.QImage(
            slice_norm.tobytes(),
            width,
            height,
            width,  # bytes por línea
            QtGui.QImage.Format_Grayscale8
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        self.image_label.setPixmap(pixmap)
        self.slice_info.setText(f"Frame (slice axial): {index + 1} / {self.volume.shape[2]}")

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = MyWidget()
    widget.resize(500, 500)
    widget.show()
    sys.exit(app.exec())