#### Stage 1 training #####
python train_stage1.py --exp_id v2 --features_type daily --n_folds 10 --backbone seresnext26t_32x4d --n_epochs 10 --batch_size 64 --init_lr 5e-4 --eta_min 3e-6
python train_stage1.py --exp_id v9 --features_type weekly --n_folds 10 --backbone seresnext26t_32x4d --n_epochs 10 --batch_size 64 --init_lr 5e-4 --eta_min 3e-6
#### Stage 2 training #####
python train_stage2.py --exp_id v2 --target_type com --features_type weekly --label_smoothing 0.1 --n_folds 10  --backbone seresnext26t_32x4d --n_epochs 40 --batch_size 64 --init_lr 5e-4 --eta_min 3e-6
python train_stage2.py --exp_id v2 --target_type res --features_type weekly --label_smoothing 0.1 --n_folds 10  --backbone seresnext26t_32x4d --n_epochs 40 --batch_size 64 --init_lr 5e-4 --eta_min 3e-6
#### Stage 3 training #####
python train_stage3.py --exp_id v4 --ckpt stage2_v2 --target_type com --features_type weekly --label_smoothing 0.1 --n_folds 10 --backbone seresnext26t_32x4d --n_epochs 40 --batch_size 64 --init_lr 5e-4 --eta_min 3e-6
python train_stage3.py --exp_id v2 --ckpt stage2_v2 --target_type res --features_type weekly --label_smoothing 0.1 --n_folds 10 --backbone seresnext26t_32x4d --n_epochs 30 --batch_size 64 --init_lr 4e-4 --eta_min 3e-6