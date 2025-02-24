CUDA_VISIBLE_DEVICES=0 python infer_shard.py --shard 0_13 &
CUDA_VISIBLE_DEVICES=1 python infer_shard.py --shard 13_26 &
CUDA_VISIBLE_DEVICES=2 python infer_shard.py --shard 26_39 &
CUDA_VISIBLE_DEVICES=3 python infer_shard.py --shard 39_50 &

wait