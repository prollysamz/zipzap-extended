set -eu

# Label-free self-supervised workflow. It deliberately does not read the
# phishing-address list or create a supervised split.
python gen_seq.py --dataset=1M --phisher=False
python gen_pretrain_data.py --do_embed=True
python partitioning.py --vocab=data/vocab --config=zipzap_config.json --output-config=data/zipzap_config.json
python run_pretrain.py --bert_config_file=data/zipzap_config.json --checkpointDir=ckpt_pretrain

# After selecting a saved checkpoint prefix, for example ckpt_pretrain/model_8001:
# python extract_embeddings.py --checkpoint=ckpt_pretrain/model_8001 --config=data/zipzap_config.json \
#   --input-file='data/embed-*.tfrecord' --output=project_results/unlabelled_embeddings.npz
# python anomaly_detection.py --train=project_results/unlabelled_embeddings.npz \
#   --method=isolation_forest --output-dir=project_results/isolation_forest
