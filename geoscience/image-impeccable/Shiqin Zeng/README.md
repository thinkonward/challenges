# Image Impeccable: Journey to Clarity

Official PyTorch implementation.<br>
3D Unet for 3D Seismic Data Denoising<br>

## Requirements

Python libraries: See [environment.yaml](environment.yaml) for library dependencies. The conda environment can be set up using these commands:

```.bash
conda env create -f environment.yaml 
conda activate seismic_Denoising
```

## Data Preparation
Open the [test.ipynb](test.ipynb) and follow the instructions to download the dataset and transfer the dataset to ``.h5`` format.
```.bash
!python data_prep/data_download.py
!python data_prep/data_format.py
```

## Train 3D Unet Models
Our training script is derived from [Deep Learning Semantic Segmentation for High-Resolution Medical Volumes](https://ieeexplore.ieee.org/abstract/document/9425041) and implemented based on [Accurate and Versatile 3D Segmentation of Plant Tissues at Cellular Resolution](https://doi.org/10.7554/eLife.57613). The training loss includes an edge loss component based on the Laplacian operator, implemented according to the paper [Multi-Stage Progressive Image Restoration](https://doi.org/10.48550/arXiv.2102.02808).


We are using one HDF5 file for training one epoch to test the code (num_epochs = 1, start = 1, end = 1).
The `start` and `end` values correspond to the dataset file names. 
For example:
`start = 1` and `end = 2` means the script will use the files `original_image-impeccable-train-data-part1.h5` and `original_image-impeccable-train-data-part2.h5`. To include all dataset files, set `start = 1` and `end = 17` to use all training data from `original_image-impeccable-train-data-part1.h5` to `original_image-impeccable-train-data-part17.h5`.
 You can modify the [config.yaml](scripts/config.yaml) file to adjust parameters such as batch_size, num_epochs, start, and end. Once you have downloaded all the data in the h5py files, set the appropriate start and end values to
train on the full dataset by running the Python script.
```.bash
!python scripts/train_model.py
```

## Test the pretrained model

The pre-trained model is in the directory [pretrained_model](pretrained_model). See the details in the [test.ipynb](test.ipynb).



