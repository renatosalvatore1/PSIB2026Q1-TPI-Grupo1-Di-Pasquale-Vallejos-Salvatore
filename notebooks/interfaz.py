import sys
import os
from PySide6 import QtCore, QtWidgets, QtGui
import numpy as np
import nibabel as nib

from prueba_final_lu import total, atlas, analisis_texturas

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        #   ---Widgets---
        self.button = QtWidgets.QPushButton("Seleccionar archivo .nii")
        
        self.title = QtWidgets.QLabel("Interfaz Diagnóstico de Tumor Cerebral y Localización", alignment=QtCore.Qt.AlignCenter)
        
        #Visualización de .nii
        self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: black;")

        #Localización con atlas
        self.info_box = QtWidgets.QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumSize(250, 400)
        self.info_box.setPlaceholderText("La localización del tumor aparecerá aquí al cargar un archivo...")
        self.info_box.setStyleSheet("background-color: #1e1e1e; color: white; font-size: 14px; padding: 10px;")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)

        self.slice_info = QtWidgets.QLabel("Frame: -", alignment=QtCore.Qt.AlignCenter)

        #   ---Layout---
        self.content_layout = QtWidgets.QHBoxLayout()
        self.content_layout.addWidget(self.image_label, stretch=2)
        self.content_layout.addWidget(self.info_box, stretch=1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.button)
        self.layout.addLayout(self.content_layout)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.slice_info)
        
        #   ---Conexiones---
        self.button.clicked.connect(self.select_file)
        self.slider.valueChanged.connect(self.update_slice)

        self.volume = None      #Guarda el array 3D del .nii
        self.file_path = None   #Guarda el path como atributo en load_nii para poder accederlo desde update_slice    

    def select_file(self):
        starting_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo .nii",
            starting_dir,
            "Todos los archivos (*)"
        )

        if file_path:
            self.load_nii(file_path)
    
    def load_nii(self, file_path):
        self.file_path = file_path
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
        slice_tumor, _ = total(slice_data) #devuelve [height, weight, 3]

        #~~~~~~~~~~~~
        #VISUALIACIÓN
        #~~~~~~~~~~~~

        # Normalizar a 0-255 para visualización
        slice_min, slice_max = slice_tumor.min(), slice_tumor.max()
        if slice_max > slice_min:
            slice_norm = (slice_tumor - slice_min) / (slice_max - slice_min) * 255
        else:
            slice_norm = np.zeros_like(slice_tumor)
        slice_norm = slice_norm.astype(np.uint8)

        # Rotar para orientación correcta (los .nii suelen venir transpuestos)
        slice_norm = np.rot90(slice_norm)

        # Manejar RGC enves de QtGui.QImage.Format_Grayscale8
        height, width, channels = slice_norm.shape
        bytes_per_line = width * channels #3 bytes por píxel en RGB

        q_image = QtGui.QImage(
            slice_norm.tobytes(),
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        self.image_label.setPixmap(pixmap)
        self.slice_info.setText(f"Frame (slice axial): {index + 1} / {self.volume.shape[2]}")

        #~~~~~~~~~~~~
        #   ATLAS
        #~~~~~~~~~~~~

        mascara_2D = slice_tumor[:,:,0] #solo canal R ~ máscara binaria
        _, prop = analisis_texturas(slice_data,mascara_2D)
        zona = atlas(prop.centroid[0], prop.centroid[1], index, self.file_path)
        self.info_box.setText(
            f"Localización del tumor:\n{zona}"
        )

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = MyWidget()
    widget.resize(900, 600)
    widget.show()
    sys.exit(app.exec())