from nilearn import datasets
from nilearn.image import load_img
import numpy as np
import nibabel as nib

def atlas(fila_tumor, columna_tumor,numero_de_corte):
    #atlas = datasets.fetch_atlas_aal()
    atlas_cort = datasets.fetch_atlas_destrieux_2009() #atlas cortical
    atlas_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm') #subcortical


    atlas_cort_img  = load_img(atlas_cort.maps)
    atlas_sub_img   = load_img(atlas_sub.maps)
    data = nib.load("../data/sub-KA02/anat/sub-KA02_run-02_T1w.nii.gz")


    paciente_affine = data.affine
    atlas_affine = atlas_sub_img.affine
    atlas_affine_inv = np.linalg.inv(atlas_affine)
    atlas_cort_data = atlas_cort_img.get_fdata()
    atlas_sub_data  = atlas_sub_img.get_fdata()
    labels = atlas_sub.labels
    voxel_tumor = np.array([fila_tumor, columna_tumor, numero_de_corte,1])
    voxel_tumor_MNI = paciente_affine @ voxel_tumor
    voxel_atlas = atlas_affine_inv @ voxel_tumor_MNI #esto tiene las coordenadas del atlas para nuestra region

    ax = int(voxel_atlas[0])
    ay = int(voxel_atlas[1])
    az = int(voxel_atlas[2])

    numero_region = int(atlas_sub_data[ax, ay, az])
    nombre_region = labels[numero_region]
    print(nombre_region)

    return nombre_region