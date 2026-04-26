This dataset includes raw and processed MRI data from a multi-center study of primary glioblastoma, collected across eight collaborating hospitals.
Data were acquired between 2018 and 2025. Participants were adults with histologically confirmed primary glioblastoma who underwent preoperative neuroimaging before surgery.

The dataset contains T1-weighted (pre- and post-contrast), T2-weighted, and FLAIR MRI sequences for all subjects. Anatomical scans were defaced using PyDeface. Processed images include skull-stripped and MNI152-registered versions of each modality. Tumor segmentation masks were generated using the SegResNet CNN model following the BraTS labeling convention and subsequently reviewed / corrected by neuroradiologists.

Raw MRI data are provided for 337 patients, organized in BIDS folders labeled sub-/site-code/-000. 
In each case, sub-XXX_run-01_T1w.nii.gz file represents pre-contrast T1-WI, and sub-XXX_run-02_T1w.nii.gz - T1-WI series obtained after Gd injection.

Derivatives/processing/sub-000 includes preprocessed structural images, such as skull-stripped T1-WI, T2-WI, and FLAIR scans registered to the MNI152 standard space.

Derivatives/processing_MNI152_skull-stripped_N4/sub-000 contains preprocessed structural MRI data (all modalities), skull-stripped and registered to the MNI152 template, with intensity non-uniformity correction performed using N4 bias field correction (AFNI).

Derivatives/processing_MNI152_skull-stripped_N4_z-scored/sub-000 includes preprocessed structural MRI data (all modalities), skull-stripped and registered to the MNI152 template, with N4 bias field correction (AFNI) followed by z-score intensity normalization (FSL). This directory represents the fully normalized, radiomics-ready variant of the dataset. 

Derivatives/segmentation/sub-*** contains the tumor segmentation masks, which correspond spatially to the preprocessed images.

Molecular data, including MGMT promoter methylation status obtained via methylation-specific PCR, and demographic information for each participant are available in the "participants.tsv" file.

List of centers participating in dataset collection and preparation:

•	Department of Biomedical and Neuromotor Sciences (DIBINEM), Bologna, Italy (site code = BO, 172 cases)
•	Städtisches Klinikum Karlsruhe, Karlsruhe, Germany (site code = KA, 80 cases)
•	Federal Neurosurgery Center, Novosibirsk, Russia (site code = NSK, 41 cases)
•	“San Paolo” Hospital, Bari, Italy (site code = BA, 3 cases)
•	Policlinico “Riuniti” Foggia, Foggia, Italy (site code = FO, 9 cases)
•	Hospital Joan XXIII, Tarragona, Spain (site code = JS, 12 cases)
•	Department of Neurosurgery, University of Turin, Turin, Italy (site code = TO, 14 cases)
•	Neurosurgery Unit, University Hospital “G. Martino”, Messina, Italy (site code = ME, 7 cases)