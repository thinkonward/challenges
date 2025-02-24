# Dark Side of the Volume – Final Solution

[**Challenge Link**](https://thinkonward.com/app/c/challenges/dark-side)

This repository contains my final solution for the **Dark Side of the Volume** seismic fault segmentation challenge. The objective of this competition is to identify faults within seismic volumes and construct 3D polygons around them. The dataset comprises 3D seismic volumes and corresponding fault masks, with the evaluation based on a 3D Dice Coefficient.

---
## 1. Installation

To set up the environment, run the following command from the repository's root directory:

```bash
sh install.sh
```

## 2. Reproducing the Final Inference

My final public score is **0.953146**, achieved using **3 UNet++ models with EfficientNet-B8 encoders**. Training each model took approximately **48 hours** on a single RTX 4090 GPU.

I used a **2D UNet++** model from the `segmentation_models_pytorch` library, with an **EfficientNet-B8** encoder pretrained on ImageNet. The solution's success is attributed to several key technical choices:

### Model Architecture
- **2.5D Architecture:** The model predicts the current frame using 7 frames simultaneously (3 previous frames, current frame, and 3 following frames), enhancing spatial consistency and model robustness.
- **Multi-axis Training:** Each model is trained on both x and y axes:
  - x-axis with frame shape (300, 1259)
  - y-axis with frame shape (300, 1259)
  This dual-axis approach enriches the model's fault detection capabilities.

### Training Strategy
- **Data Augmentation:** Applied transformations including 180° rotation and horizontal/vertical flips to improve model generalization.
- **Model Ensembling:** Dataset split into 6 folds (325 training volumes, 65 validation volumes per fold). Final solution uses models from the first 3 folds, each trained for up to 12 epochs.

Even one model UNet++ with EfficientNet-B8 on fold 0 reaches ~0.947 on public leaderboard, demonstrating the effectiveness of the architecture.

Inference time for a single model on one axis takes around 5 minutes, so 5 * 2 axes * 3 models = ~30 minutes total inference time on a single RTX 4090 GPU.

There are two ways to reproduce the inference steps:

### 2.1. Using the Notebook

Follow the instructions in [`solution_pipeline.ipynb`](solution_pipeline.ipynb). This notebook includes:

- **Loading Trained Model Checkpoints:** Instructions to load the saved model weights.
- **Generating 3D Predictions:** Steps to create fault predictions for the test dataset.
- **Creating a Submission File:** Process to compile predictions into a `.npz` file suitable for submission.

You have to change the value of the 2 variables ROOT_PATH_TO_ALL_TEST_PARTS and PATH_TO_2D_SLICES_OUTPUT

### 2.2. Using Command Line

Alternatively, you can reproduce the inference using command-line instructions:

1. First, generate 2D slices from the test volumes:

```bash
python src/write_2d_slices.py \
  --root-dir ROOT_PATH_TO_ALL_TEST_PARTS \
  --output-dir PATH_TO_2D_SLICES_OUTPUT \
  --axes x y \
  --mode test \
  --num-workers 25
```

2. Then, run the prediction using the trained model checkpoints:

```bash
python predict.py PATH_TO_2D_SLICES_OUTPUT \
  --checkpoints checkpoints/checkpoints_unetpp_timm-efficientnet-b8_nchans7_val_axisxy_20250102_152812/fold0-best-model-epoch=09-val_dice_3d=0.8959.ckpt \
    checkpoints/checkpoints_unetpp_timm-efficientnet-b8_nchans7_val_axisxy_20250102_152812/fold1-best-model-epoch=10-val_dice_3d=0.8988.ckpt \
    checkpoints/checkpoints_unetpp_timm-efficientnet-b8_nchans7_val_axisxy_20250102_152812/fold2-best-model-epoch=10-val_dice_3d=0.8870.ckpt \
  --axes xy xy xy \
  --num_workers 4 \
  --batch_size 4 \
  --min_mean_conf 0.1 \
  --save_proba
```

**Note:** To clean the cache (probability volumes from each model on each axis) written by the prediction script, delete the `predictions_probas` folder in the corresponding checkpoint directory.

## 3. Reproducing the Final Training

To replicate the training process of my final solution, follow these steps:

### 3.1. Generating 2D Slices

First, convert the 3D seismic volumes into 2D slices suitable for training:

```bash
python src/write_2d_slices.py \
  --root-dir ROOT_PATH_TO_ALL_TRAIN_PARTS \
  --output-dir PATH_TO_2D_SLICES_OUTPUT \
  --axes x y \
  --mode train \
  --num-workers 25
```

Parameters:

--root-dir: Path to the directory containing all training volumes.

--output-dir: Destination directory for the generated 2D slices.

--axes x y: Generate slices along both the x and y axes.

--num-workers 36: Number of worker threads for data processing.

Additional Note: If ROOT_PATH_TO_ALL_TRAIN_PARTS includes test volumes, add the "--mode 'train'" argument to ensure only training volumes are processed.

### 3.2. Training the Model

Once the 2D slices are generated, initiate the training process with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py PATH_TO_2D_SLICES_OUTPUT \
  --num_workers 4 \
  --archi unetpp \
  --encoder_name timm-efficientnet-b8 \
  --epochs 12 \
  --seed 1023 \
  --nchans 7 \
  --early_stop 3 \
  --nfolds 6 \
  --train_axis xy \
  --val_axis xy \
  --aug \
  --n_bins 40 \
  --excluded_vol 2023-10-05_37fd5dd2 2023-10-05_aa1525c4 2023-10-05_407bcfc6 \
                2023-10-05_795448d4 2023-10-05_74a447f1 2023-10-05_ba087996 \
                2023-10-05_c923057a 2023-10-05_eff8c5d7 \
  --batch_size 2 \
  --lr 1e-4 \
  --scheduler_gamma 0.8 \
  --dropout 0.5
```

What This Command Does:

**Data Preparation**: Excludes volumes with zero faults and prepares data for training.

**Model Training**: Trains the UNet++ model with the specified parameters across 6 folds. Due to time constraints, my final solution utilizes only the models from the first 3 folds.

**Checkpoint Saving**: Saves model checkpoints after each fold, which are later used for ensembling during inference.

You can try to increase the number of epochs to improve final score.

**Note about multi-GPU training:** You can try training with multiple GPUs (by removing `CUDA_VISIBLE_DEVICES=0` from the start of the command), though this didn't work in my testing environment.

**Note about checkpoint size:** If you need to reduce the size of model checkpoints (e.g., for sharing or storage purposes), you can use the provided conversion script:

```bash
python scripts/checkpoint_converter.py PATH_TO_CKPT
```

### 4. Bonus

### Training with U-Mamba

I also trained a larger model using the U-Mamba architecture (from https://github.com/bowang-lab/U-Mamba/tree/main). I incorporated the authors' code into my final solution under the umamba directory.

This model alone achieves a score of 0.953543 on the public leaderboard using a single fold. However, the installation of U-Mamba is more complex, which is why it wasn't included in the final submission. Nevertheless, you can install it using the provided script:

```bash
sh install_with_umamba.sh
```

Note: This setup has been tested and works in my local environment and on Kaggle T4x2 notebooks.

### Reproducing the Training with U-Mamba

If the installation of U-Mamba was successful, you can reproduce the training with the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py PATH_TO_2D_SLICES_OUTPUT \
  --num_workers 4 \
  --archi umamba \
  --model_size xxxlarge \
  --epochs 20 \
  --seed 1023 \
  --nchans 9 \
  --early_stop 8 \
  --nfolds 8 \
  --train_axis xy \
  --val_axis xy \
  --aug \
  --n_bins 40 \
  --excluded_vol 2023-10-05_37fd5dd2 2023-10-05_aa1525c4 2023-10-05_407bcfc6 \
                2023-10-05_795448d4 2023-10-05_74a447f1 2023-10-05_ba087996 \
                2023-10-05_c923057a 2023-10-05_eff8c5d7 \
  --batch_size 2 \
  --lr 1e-4 \
  --scheduler_gamma 0.8 \
  --dropout 0.5
```

**Training Time:**

Training a single fold with this configuration takes approximately **7 days** on an RTX 4090 GPU (about **16 hours per epoch**).

**Best Approach:**

If training/inference time were not a constraint, the best solution would involve ensembling multiple U-Mamba models across different folds to achieve even higher scores.

### Reproducing the Inference with U-Mamba

To reproduce the inference using the U-Mamba model, use the [`solution_pipeline.ipynb`](solution_pipeline.ipynb) notebook and select the specific checkpoint:

checkpoints_umamba_xxxlarge_nchans9_val_axisxy_20241226_004424/fold0-best-model-epoch=05-val_dice_3d=0.8899.ckpt

This model can predict all 50 test volumes on both x and y axes in approximately **35 minutes** on a single RTX 4090 GPU.

## 5. Contact

If you encounter any issues reproducing my solution or have any questions, feel free to reach out:

## 6. Acknowledgments

- **Challenge Organizers:** Onward Challenges for hosting the Dark Side of the Volume competition.
- **Data Providers:** Thanks to the official repository for providing the necessary data and references.
- **UNet++ Authors:** Gratitude to the authors of the [UNet++ paper](https://arxiv.org/pdf/1807.10165) for sharing their architecture.
- **U-Mamba Authors:** Appreciation to the developers of [U-Mamba](https://github.com/bowang-lab/U-Mamba) for their work.
