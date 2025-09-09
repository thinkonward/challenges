# Short Description 
This project is focused on implementing a seismic velocity model prediction system using multi-sources seismic input data processed with a Transformer-based EVA architecture. The model employs advanced multi-head attention mechanisms to effectively fuse information across multiple seismic sources, enhancing prediction accuracy.

Key features include:

- Implementing a custom PyTorch Dataset for loading and preprocessing multi-source seismic data and corresponding velocity models.

- Implementing channel fusion using dual and single multi-head attention modules to capture inter-channel dependencies.

- Implementing a comprehensive training and validation pipeline with support for mixed precision training, learning rate scheduling, and early stopping.

- Implementing test-time augmentation using horizontal flips to improve inference robustness.

- Supporting both training/validation and test modes, with flexible data augmentation.

This project aims to implement an efficient and accurate approach for seismic velocity inversion, crucial for subsurface characterization in geophysical applications.

# Installation 
1. Create a new conda environment and install the required packages:
```bash
conda create -n onward python=3.11.7
conda activate onward

```
2. Install requiremnts.txt:
```

# Project Structure
The repository is structured as follows:
```
.
├── README.md
├── Solution_pipeline.ipynb
├── Submissions/
├── data/
│   ├── test.csv
│   ├── test_data
│   ├── train_data
│   ├── train_extended_1
│   ├── train_extended_2
│   ├── train_fold0.csv
│   ├── train_fold1.csv
│   ├── train_fold2.csv
│   ├── train_fold3.csv
│   ├── train_fold4.csv
│   ├── val_fold0.csv
│   ├── val_fold1.csv
│   ├── val_fold2.csv
│   ├── val_fold3.csv
│   └── val_fold4.csv
├── model_folds.yaml
├── predict.py
├── requirements.txt
├── run_training.py
├── speed_structure_checkpoints/
│   ├── Checkpoints
│   └── __init__.py
├── src/
│   ├── __init__.py
│   ├── __pycache__
│   ├── dataset.py
│   ├── engine.py
│   ├── models
│   ├── muon.py
│   ├── prepare_data-test.py
│   ├── prepare_data_train.py
│   ├── train_EVA_16_Large_Split_10_Multi_MHA_2_heads.py
│   ├── train_EVA_16_Large_Split_10_Multi_MHA_4_heads.py
│   ├── train_EVA_16_Large_Split_10_Single_MHA_4_heads.py
│   ├── train_EVA_16_Large_Split_9_Multi_MHA_4_heads.py
│   └── utils.py
​
```

- `README.md`: This file provides an overview and explanation of the project structure, usage, and instructions.

- `Solution_pipeline.ipynb`: A Jupyter notebook containing a detailed solution report, including analysis, methodology, and results with ready to run training and inference

- `Submissions/`: Directory to store trained model weights for each fold and model configuration during training. Also generated .npz file after inference.

- `data/`: Contains dataset folders. The following datasets should be unzipped here into:
        train_data
        train_extended_1
        train_extended_2
        test_data
    CSV files for each fold are generated according to the procedure in Solution_pipeline.ipynb.

- `model_folds.yaml`:Configuration file specifying fold assignments for each model. Used by run_training.py to automate training across folds and models.

- `predict.py`: Script for generating predictions on the test dataset using trained models.

- `requirements.txt`: Lists required Python packages and their specific versions for environment setup.

- `run_training.py`: Script to automate the full training process by iterating through all folds and relevant training scripts (train_EVA_16_***.py) based on fold assignments in model_folds.yaml.

- `speed_structure_checkpoints/`: Contains trained model checkpoints under the subfolder Checkpoints/ used for inference.

- `src/`: Source code directory with various modules:

- `Dataset.py`: Dataset routines for loading seismic data and seismic velocity, converting to tensors for training and inference.

- `engine.py`: Handles training and validation logic for one epoch.

- `models`: Contains model definitions and handlers, including EVA 16 models with Multi-Head Attention.
        model_EVA_16_Large_Split_10_Multi_MHA_2_heads.py
        model_EVA_16_Large_Split_10_Multi_MHA_4_heads.py
        model_EVA_16_Large_Split_10_Single_MHA_4_heads.py
        model_EVA_16_Large_Split_9_Multi_MHA_4_heads.py
        model_EVA_MHA_Handler.py
        
- `muon.py`: Implementation of the Muon optimizer.

- `prepare_data-test.py`: Script to generate CSV files for the test dataset, required by dataset and dataloader scripts.

- `prepare_data-train.py`: Executes K-Means clustering on training data to identify seismic groups, then creates K-Fold CSV files for training and validation folds (e.g., train_fold0.csv, val_fold0.csv). Required by dataset and dataloader scripts.

- `Training scripts (train_EVA_16_*.py)`: Each script runs training for a specific model configuration and fold. These are called with command-line arguments specifying the fold, data directory (absolute path), and checkpoint save path. Manual runs are not required as run_training.py automates this process.
        train_EVA_16_Large_Split_10_Multi_MHA_2_heads.py
        train_EVA_16_Large_Split_10_Multi_MHA_4_heads.py
        train_EVA_16_Large_Split_10_Single_MHA_4_heads.py
        train_EVA_16_Large_Split_9_Multi_MHA_4_heads.py
        
- `utils.py`: General utility functions used across the project.


# Solution Reproduction Note
To understand and reproduce my solution, please refer to the `solution.ipynb` notebook. The notebook provides a detailed explanation of the methodology, implementation, and some insights.

# Runtime Details:
Training each fold per model takes around 13hours for the full epochs to be completed A6000. There are 3 models per 5 folds, requiring 15 model runs
Inference takes 9 minutes to inference the 3 models per 5 folds, ensemble and make final prediction (A6000).

# Source Processing Improvements
In order to show improvements with certain architetcure changes, I provide a table with the experiments conducted during the development of the solution. The table includes the Experiment Description and public MAPE score.

| Experiment Id | Description | Public MAPE Score| Checkpoints|
| --- | --- | --- |
| 1 | One Fold EVA Large Split 10 + Mean average pooling channel (No attention)| 0.0246 - 0.0248 | NIL|
| 2 | One Fold EVA Large Split 10 + Channel Multihead Attention(4 heads) + Mean Fusion| 0.02408 - 0.02419 |EVA_16_Large_Split_10_Single_MHA_4_heads|
| 3 | One Fold EVA Large Split 10 + Channel Multihead Attention(4 heads) + Fusion Multihead Attention(4 heads) |0.02391 - 0.02398 |EVA_16_Large_Split_10_Dual_MHA_4_heads|
| 4 | 5 Folds EVA Large Split 10 + Channel Multihead Attention(4 heads) + Fusion Multihead Attention(4 heads) |0.023491|EVA_16_Large_Split_10_Dual_MHA_4_heads|
| 5 | 5 Folds Ensemble of 2, 3 + One additional configuration of Channel Multihead Fusion |0.023458|EVA_16_Large_Split_10_Single_MHA_4_heads, EVA_16_Large_Split_10_Dual_MHA_4_heads, EVA_16_Large_Split_10_Dual_MHA_2_heads, EVA_16_Large_Split_9_Dual_MHA_4_heads|