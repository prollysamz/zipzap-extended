set -eu

python gen_seq.py --dataset=10M # construct transaction sequence
python gen_pretrain_data.py # generate pre-training data
python gen_finetune_data.py # generate fine-tuning data

python run_pretrain.py --checkpointDir=ckpt_pretrain # pre-training
python run_finetune.py --init_checkpoint=ckpt_pretrain # fine-tuning and validation-calibrated evaluation
