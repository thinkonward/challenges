for seed in  7 100 700 1000 70000  ;do
  

    export CUDA_VISIBLE_DEVICES=0,1
    model_name=exp149_all${seed}
    
    python -m run.train_dist \
      dataset.num_folds=-1 \
      dataset.test_fold=0 \
      dataset.phase=train \
      dataset.meta_path=data/meta_all.csv \
      model.name=model_v37\
      model.params.image_size=[624,624] \
      model.params.patch_size=16 \
      +model.params.split_at=8 \
      +model.params.fusion_method=weighted\
      model.params.base_model=timm/eva02_base_patch16_clip_224 \
      forwarder.mix.type=null \
      forwarder.loss.l1_weight=0. \
      +forwarder.loss.hybrid_weight=1. \
      training.seed=${seed} \
      training.batch_size=12\
      training.batch_size_test=12\
      training.accumulate_grad_batches=1 \
      training.epoch=15\
      training.num_workers=8 \
      training.num_gpus=2 \
      optimizer.lr=4e-4 \
      scheduler.type=constant_cosine \
      scheduler.lr_decay_scale=0.1 \
      scheduler.warmup_steps_ratio=0.5 \
      training.use_wandb=true \
      training.sync_batchnorm=false \
      training.use_amp=true \
      optimizer.type=muon \
      optimizer.weight_decay=0.01 \
      training.use_gradient_checkpointing=false \
      out_dir=results/${model_name}

    mkdir inference_models/${model_name}
    cp $(find results/${model_name}/weights -name model_weights.pth) inference_models/${model_name}/model_weights.pth
    cp $(find results/${model_name}/weights -name model_weights_ema.pth) inference_models/${model_name}/model_weights_ema.pth
    cp results/${model_name}/.hydra/config.yaml inference_models/${model_name}/
  
done
