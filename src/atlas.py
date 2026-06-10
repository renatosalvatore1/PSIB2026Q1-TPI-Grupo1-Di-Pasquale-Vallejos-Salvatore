from nilearn import datasets
from nilearn.image import load_img
import numpy as np
import nibabel as nib

def atlas(fila_tumor, columna_tumor,numero_de_corte,file_path,tumor_detectado):
    if tumor_detectado == 1:
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm') #subcortical
        atlas_img  = load_img(atlas.maps)
        data = nib.load(file_path)
        paciente_affine = data.affine
        atlas_affine = atlas_img.affine
        atlas_affine_inv = np.linalg.inv(atlas_affine)
        atlas_data  = atlas_img.get_fdata()
        labels = atlas.labels
        voxel_tumor = np.array([fila_tumor, columna_tumor, numero_de_corte,1])
        voxel_tumor_MNI = paciente_affine @ voxel_tumor
        voxel_atlas = atlas_affine_inv @ voxel_tumor_MNI #esto tiene las coordenadas del atlas para nuestra region

        ax = int(voxel_atlas[0])
        ay = int(voxel_atlas[1])
        az = int(voxel_atlas[2])

        numero_region = int(atlas_data[ax, ay, az])
        nombre_region = labels[numero_region]

        if nombre_region == 'Background':
            atlas = datasets.fetch_atlas_destrieux_2009() #atlas cortical
            atlas_img  = load_img(atlas.maps)
            data = nib.load(file_path)
            paciente_affine = data.affine
            atlas_affine = atlas_img.affine
            atlas_affine_inv = np.linalg.inv(atlas_affine)
            atlas_data  = atlas_img.get_fdata()
            labels = atlas.labels
            voxel_tumor = np.array([fila_tumor, columna_tumor, numero_de_corte,1])
            voxel_tumor_MNI = paciente_affine @ voxel_tumor
            voxel_atlas = atlas_affine_inv @ voxel_tumor_MNI #esto tiene las coordenadas del atlas para nuestra region

            ax = int(voxel_atlas[0])
            ay = int(voxel_atlas[1])
            az = int(voxel_atlas[2])

            numero_region = int(atlas_data[ax, ay, az])
            nombre_region = labels[numero_region]

            if nombre_region == 'Background':
                nombre_region = "Region no indexada por el atlas."
    else:
        nombre_region = "No se detectaron tumores."

    return nombre_region