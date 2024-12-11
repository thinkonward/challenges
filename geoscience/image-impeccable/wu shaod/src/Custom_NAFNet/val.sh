# python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/SIDD/NAFNet-width32.yml --launcher pytorch

python basicsr/test.py -opt options/train/Custom/CUSTOM_NAFNet-width32.yml