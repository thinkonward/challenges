model=[
    dict(
        type = "SMPUnetModel",
        encoder=dict(
            encoder_name    = 'tu-tf_efficientnetv2_s',
            in_channels     = 3,
            encoder_depth   = 5,
            encoder_weights = None,
        ),
        decoder=dict(
            decoder_name            = 'UnetDecoder',
            decoder_channels        = (256, 128, 64, 32, 16),
            decoder_use_batchnorm   = False,
            decoder_attention_type  = None,
            num_classes             = 1,
            align_corners           = False
        ),
        pretrained_path     = r"./pretrained_model/tf_efficientnetv2_s_21k-6337ad01.pth",
    ),

    dict(
        type = "SMPUnetModel",
        encoder=dict(
            encoder_name    = 'tu-tf_efficientnetv2_s',
            in_channels     = 3,
            encoder_depth   = 5,
            encoder_weights = None,
        ),
        decoder=dict(
            decoder_name            = 'UnetDecoder',
            decoder_channels        = (256, 128, 64, 32, 16),
            decoder_use_batchnorm   = False,
            decoder_attention_type  = None,
            num_classes             = 1,
            align_corners           = False
        ),
        pretrained_path     = r"./pretrained_model/tf_efficientnetv2_s_21k-6337ad01.pth",
    ),
]

loss2=dict(type="SMP_BCE", 
            weight=None,
            ignore_index=None,
            reduction="mean",
            smooth_factor=None,
            pos_weight=None,
)


loss=dict(type="SMP_DICE", 
            mode="binary",
            from_logits=True,
)

dataset=dict(
    train=dict(type="Train_25D_Dataset",
               near_slice_shuffle=False,
                root_dir="./data/train_data_normed/",
                txt_file="./data/train_txt/train_f0.txt"
    ),
    val=dict(type="ValDataset",
                root_dir="./data/train_data_normed/",
                txt_file="./data/train_txt/val_f0.txt"
    )
)
optimizer=dict(type="Adam", lr=1e-3)
scheduler=dict(type="PolyLR")
train_cfg=dict(use_amp=False,
               use_swa=True,
               swa_start_epc=30,
               swa_lr=3e-4,
                batch_size=24,
                num_workers=4,
                save_path="./experiments/exp_f0/",
                train_epochs=40,
                val_iterations=4000,
                log_step=100,
                seed=3407,
)
val_cfg=dict(pad_to_32=True, 
             pos_thr=0.5,
             test_dim_tta=["h",],
             test_flip_tta=["normal"],
)

test_cfg=dict(pad_to_32=True,
                pos_thr=0.5,
                test_dim_tta=["h", "w"],
                test_flip_tta=["normal", "h_flip_tta", "v_flip_tta"],
)
device="cuda:0"