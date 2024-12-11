# Journey

I started the comp pretty late, the first day was on October 1 and my goal was to make a sub asap so I gave it a couple of hours, so the real first day was October 11 where I made the baseline score 0.981 on LB, over the next few days I did a lot of reading and trying out ideas, so here is how they went:

### Research on finding image to image models
Diffusion models were very expensive to train so I couldn't train them much, SuperResolution models that used CNN was my best bet because I did not have enough compute and score scaled with more compute and more training time, I tried out NAFNet, I got it to score really decent at 0.985, I did not train more of it because it was very expensive computation wise

### Discovery of the decoder bottleneck
Something just felt off for 2 days, it seemed like some bottleneck, so I did a training where the input and target is the same, its both the same image, I don't resize my images I only zero pad them to 320x1280 resolution so the question I wanted the answer to was can models even preserve the input pixel level information? The answer was NO!, atleast not for Unet, I switched the Unet interpolation to "PixelShuffle" which preserves the pixel level information and I got an SSIM of close to 1 (note that input and target is the same image in this example)

I tried with UnetPlusPlus and because there were a lot more connections in that decoder, it was better than Unet, I trained them all and found out: UnetPlusPlus and PixelShuffleUnet did pretty similar (dissappointed that Pixel Shuffle didn't increase the score a lot) so I ran my last training with UnetPlusPlus

### Where did it go wrong?
I don't know if there was just a better way or I missed something big if that's not it then my hunch is that it was the compute limitation, I could only use 20% of the data (so I take 60/300 slices) but I have LB 0.9875 (and CV to support it) with 1 model and with decent synergy of efficientnet and NFNet and full data training I can see myself getting a much better score

### Why not Transformer Models?
Compute.... That's the only reason...

But to elaborate is, I have 24GB gpus, mostly enough but when you want to input 320x1280 image, not so much with transformer models, so I tried taking patches (320x320) of the image and training of them which scored worse on CV, like -0.005 so I did not go into it further due of time constraints of the last week

I do wish I would have joined another week or two before, I could have had more time to test out each idea

# Approach
My final submit of a single model 0.9875 is pretty simple, its an efficientnet-v2s encoder and unet++ decoder, I take like ~96% of the volume data for this training, with 1 slice after 4 so, 20% of the slices for each volume, I train for 100 epochs, it takes like 12 hours

### Ideas that didn't work
- Scaling noise higher or lower
- Predicting Noise instead of Denoised was not better, it was not worse either
- Training a model to predict Denoised from the predictions of the last model, acting like a further refinement model was pretty identical to longer training and no considerate boost on its own was observed
- Augmentations or Test-Time Augmentations except Vertical Flip

## Installation

Let's start with making a conda env

```bash
conda create -n impeccable_obs anaconda python=3.10
```

We activate the env with
```bash
conda activate impeccable_obs
```

Installing the libraries

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install timm segmentation_models_pytorch
pip install transformers accelerate
pip install opencv-python albumentations
pip install hydra-core
pip install neptune
pip install torch_ema
pip install zstandard redis
```

PS: I installed redis quite a while ago so I don't remember if more installation than just a pip install is needed, so follow the web/gpt-4o for that, GL

# Usage

Structure:
```
├── configs
│   ├── config.yaml
│   ├── criterion
│   │   └── MSELoss.yaml
│   ├── dataset
│   │   ├── ObsidianDataset.yaml
│   │   └── ObsidianTestDataset.yaml
│   ├── metric
│   │   ├── MSE.yaml
│   │   └── SSIM.yaml
│   ├── model
│   │   ├── SMPModel.yaml
│   │   └── TimmModel.yaml
│   ├── optimizer
│   │   └── AdamW.yaml
│   └── scheduler
│       └── CosineAnnealing.yaml
├── data
    ├── image_impeccable_starter_notebook
    └── image-impeccable-submission-sample.npz
    └── processed_test1.csv
    └── processed_train1.csv
    └── test_data
    └── test_minmax.csv
    └── train_data
    └── train_master1.csv

├── main.py
├── models
│   └── v2s_v30
│       ├── 0_best.pth
│       ├── 0_EMA.pth
│       ├── cache
│       ├── run.py
│       ├── test_config.yaml
│       ├── test_prediction_dict.npy
│       └── v2s_v30_last_flip_obs.npz
├── obsidian
│   ├── datasets
│   │   ├── ObsidianDataset.py
│   ├── __init__.py
│   ├── logging_helpers
│   │   ├── helpers.py
│   ├── metrics
│   │   ├── metrics.py
│   ├── models
│   │   ├── SMPModel.py
│   │   └── TimmModel.py
│   ├── tester
│   │   └── tester.py
│   └── trainer
│       └── trainer.py
├── predict.sh
├── README.md
└── test.py
```

First we preprocess the data:

For easy access to the volumes, I convert them to .h5 format which takes more disk storage and make a processed .csv file to help the scripts

I also calculate SSIM on denoised vs noised volumes in training and if its too low, like lower than 0.7 I don't take that volume for training

```bash
python preprocess_train.py
python preprocess_test.py
```

To predict on the test set once the preprocessing is done:

```bash
./predict.sh
```

PS: If accelerate is not working for you, you can do ```accelerate config``` and answer the questions on a multi-gpu system

To Re-Train my model you can do:
```bash
accelerate launch --main_process_port=29501 main.py data_info.fold=[0]
```