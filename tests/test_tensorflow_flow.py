"""Small real TensorFlow flow, including shared-layer loading and singleton batches."""
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow.compat.v1")
from research_protocol import read_provenance, sha256_file, write_json
from vocab import FreqVocab


def run_script(model_dir, script, *arguments, success=True):
    result = subprocess.run([sys.executable, script, *arguments], cwd=model_dir,
                            capture_output=True, text=True, timeout=120)
    if success:
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-6000:]
    else:
        assert result.returncode != 0, result.stdout
    return result


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("zipzap")
    model_dir = root / "Model"
    model_dir.mkdir()
    for file in (Path(__file__).parents[1] / "Model").glob("*.py"):
        shutil.copy(file, model_dir)
    data = model_dir / "data"
    data.mkdir()
    (root / "Data").mkdir()
    accounts = ["account_%03d" % i for i in range(80)]
    (root / "Data/phisher_account.txt").write_text("\n".join(accounts[:20]) + "\n")
    sequences = {account: [[accounts[(i + j + 1) % 80], 100 + j, 1000 + j,
                            12 - j, "OUT" if j % 2 else "IN", 1 + j]
                           for j in range(5)] for i, account in enumerate(accounts)}
    with open(data / "eoa2seq.pkl", "wb") as output:
        pickle.dump(sequences, output)
    run_script(model_dir, "gen_pretrain_data.py", "--max_seq_length=6", "--dupe_factor=1",
               "--total_drop=False", "--seed=7")
    with open(data / "vocab", "rb") as source:
        vocab = pickle.load(source)
    size = max(vocab.id_to_tokens) + 1
    config = dict(vocab_size=size, hidden_size=4, bucket_list=[[0, 10], [10, size]],
                  factor_list=[4, 2], num_hidden_layers=2, num_attention_heads=2,
                  intermediate_size=8, max_position_embeddings=10,
                  hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                  initializer_range=0.02, type_vocab_size=2)
    write_json(model_dir / "config.json", config)
    run_script(model_dir, "gen_finetune_data.py", "--max_seq_length=6", "--seed=7")
    run_script(model_dir, "run_pretrain.py", "--bert_config_file=config.json", "--max_seq_length=6",
               "--epoch=1", "--batch_size=11", "--neg_sample_num=4", "--seed=7",
               "--num_warmup_steps=0", "--num_train_steps=100")
    checkpoint = next((model_dir / "ckpt_dir").glob("model_*.index"))
    assert "cls/predictions/output_bias" not in {
        name for name, _ in tf.train.list_variables(str(checkpoint.with_suffix("")))}
    return model_dir, str(checkpoint.with_suffix(""))


def record_accounts(path):
    return {tf.train.Example.FromString(record).features.feature["address"].int64_list.value[0]
            for record in tf.io.tf_record_iterator(str(path))}


def test_default_pretraining_uses_all_accounts_without_holdout(experiment):
    model_dir, checkpoint = experiment
    data = model_dir / "data"
    manifest = json.loads((data / "pretrain_manifest.json").read_text())
    assert manifest["diagnostic_holdout"] is False and manifest["train_covers_all_accounts"] is True
    assert manifest["validation_sha256"] is None
    assert not (data / "test.tfrecord").exists() and not (data / "pretrain_split.json").exists()
    with open(data / "vocab", "rb") as source:
        vocab = pickle.load(source)
    expected = set(vocab.convert_tokens_to_ids(["account_%03d" % i for i in range(80)]))
    assert record_accounts(data / "train.tfrecord") == expected
    provenance = read_provenance(checkpoint)
    assert provenance["diagnostic_holdout"] is False and provenance["trained_on_all_accounts"] is True
    rejected = run_script(model_dir, "run_pretrain.py", "--bert_config_file=config.json", "--max_seq_length=6",
                          "--do_train=False", "--do_eval=True", "--init_checkpoint=" + checkpoint,
                          "--checkpointDir=ckpt_no_holdout", success=False)
    assert "No diagnostic holdout" in rejected.stderr


def test_diagnostic_holdout_is_account_disjoint_and_not_used_for_embeddings(experiment):
    model_dir, _ = experiment
    diag = "data_diag/"
    shutil.copytree(model_dir / "data", model_dir / "data_diag")
    run_script(model_dir, "gen_pretrain_data.py", "--max_seq_length=6", "--dupe_factor=1",
               "--total_drop=False", "--seed=7", "--do_eval=True", "--data_dir=" + diag)
    data = model_dir / "data_diag"
    manifest = json.loads((data / "pretrain_manifest.json").read_text())
    assert manifest["diagnostic_holdout"] is True and manifest["train_covers_all_accounts"] is False
    split = json.loads((data / "pretrain_split.json").read_text())
    train_accounts = record_accounts(data / "train.tfrecord")
    holdout_accounts = record_accounts(data / "test.tfrecord")
    assert holdout_accounts and not train_accounts & holdout_accounts
    assert train_accounts == set(split["train_accounts"]) and holdout_accounts == set(split["validation_accounts"])
    assert len(train_accounts) + len(holdout_accounts) == 80
    common = ["run_pretrain.py", "--bert_config_file=config.json", "--max_seq_length=6", "--epoch=1",
              "--batch_size=11", "--neg_sample_num=4", "--seed=7", "--num_warmup_steps=0",
              "--num_train_steps=100", "--data_dir=" + diag, "--checkpointDir=ckpt_diag",
              "--train_input_file=data_diag/train.tfrecord", "--test_input_file=data_diag/test.tfrecord",
              "--pretrain_manifest=data_diag/pretrain_manifest.json"]
    run_script(model_dir, *common)
    diag_checkpoint = str(next((model_dir / "ckpt_diag").glob("model_*.index")).with_suffix(""))
    provenance = read_provenance(diag_checkpoint)
    assert provenance["diagnostic_holdout"] is True and provenance["trained_on_all_accounts"] is False
    # The holdout loss can be measured from the diagnostic checkpoint.
    result = run_script(model_dir, *common, "--do_train=False", "--do_eval=True",
                        "--init_checkpoint=" + diag_checkpoint)
    assert "Evaluation Results" in result.stdout
    # Downstream embeddings must not silently use an encoder that skipped part of the accounts.
    export = ["extract_embeddings.py", "--checkpoint=" + diag_checkpoint, "--config=config.json",
              "--input-file=data/finetune_test.tfrecord", "--split=test", "--vocab=data_diag/vocab",
              "--batch-size=11"]
    rejected = run_script(model_dir, *export, "--output=project_results/diag_rejected.npz", success=False)
    assert "account holdout" in rejected.stderr
    run_script(model_dir, *export, "--allow-diagnostic-checkpoint", "--output=project_results/diag.npz")
    report = json.loads((model_dir / "project_results/diag.npz.json").read_text())
    assert report["diagnostic_holdout_checkpoint"] is True


def test_data_features_and_count_bucket_boundaries(experiment):
    model_dir, _ = experiment
    manifest = json.loads((model_dir / "data/finetune_splits.json").read_text())
    assert [len(manifest["splits"][s]["account_ids"]) for s in ("train", "validation", "test")] == [56, 12, 12]
    for split in manifest["splits"]:
        file = model_dir / ("data/finetune_" + split + ".tfrecord")
        assert sha256_file(file) == manifest["splits"][split]["sha256"]
        for record in tf.io.tf_record_iterator(str(file)):
            feature = tf.train.Example.FromString(record).features.feature
            assert list(feature["input_values"].int64_list.value) == [0, 12, 11, 10, 9, 8]
            assert list(feature["input_counts"].int64_list.value) == [0, 1, 2, 3, 4, 5]
    result = run_script(model_dir, "-c", "import gen_seq; print([gen_seq.feature_bucketization({'a': [['b',0,0,100,'OUT',n]]})['a'][0][5] for n in [7,8,9,10,11]])")
    assert "[7, 8, 8, 8, 9]" in result.stdout


def test_frozen_training_evaluation_export_and_failure_paths(experiment):
    model_dir, checkpoint = experiment
    args = ["--bert_config_file=config.json", "--max_seq_length=6", "--batch_size=11",
            "--seed=7", "--hidden_size=4", "--num_warmup_steps=0", "--num_train_steps=100"]
    run_script(model_dir, "run_finetune.py", *args, "--freeze_encoder=True",
               "--init_checkpoint=" + checkpoint, "--output_dir=project_results/frozen")
    frozen = str(model_dir / "ckpt_dir/bert_frozen_mlp")
    source = tf.train.load_checkpoint(checkpoint)
    target = tf.train.load_checkpoint(frozen)
    for name in target.get_variable_to_shape_map():
        if name.startswith("bert/"):
            source_name = name.replace("/layer_0/", "/shared_layer/").replace("/layer_1/", "/shared_layer/")
            np.testing.assert_array_equal(target.get_tensor(name), source.get_tensor(source_name))
    report = json.loads((model_dir / "project_results/frozen/metrics.json").read_text())
    assert report["accounts"] == 12  # batch size 11 must retain the final singleton
    assert report["coverage_verified"] and report["threshold_label_use"]
    assert report["checkpoint_provenance"]["encoder_label_use"] is False
    run_script(model_dir, "run_finetune.py", *args, "--freeze_encoder=True",
               "--checkpointDir=ckpt_random", "--output_dir=project_results/random")
    random_provenance = read_provenance(str(model_dir / "ckpt_random/bert_frozen_mlp"))
    assert random_provenance["encoder_initialization"] == "random"
    # Evaluation-only must not open the training TFRecord or create a train graph.
    run_script(model_dir, "run_finetune.py", *args, "--do_train=False", "--load_cross=False",
               "--init_checkpoint=" + frozen, "--train_input_file=does-not-exist.tfrecord",
               "--output_dir=project_results/eval_only")
    run_script(model_dir, "extract_embeddings.py", "--checkpoint=" + checkpoint,
               "--config=config.json", "--input-file=data/finetune_test.tfrecord", "--split=test",
               "--batch-size=11", "--output=project_results/embeddings.npz")
    with np.load(model_dir / "project_results/embeddings.npz") as export:
        assert export["embeddings"].shape == (12, 4)
        assert len(np.unique(export["account_ids"])) == 12
    run_script(model_dir, "extract_embeddings.py", "--checkpoint=" + checkpoint,
               "--config=config.json", "--input-file=data/finetune_test.tfrecord", "--split=test",
               "--pooling=mean", "--output=project_results/mean.npz")
    run_script(model_dir, "run_finetune.py", *args, "--init_checkpoint=" + checkpoint,
               "--output_dir=project_results/tuned")
    tuned = str(model_dir / "ckpt_dir/bert_finetune")
    assert read_provenance(tuned)["encoder_label_use"] is True
    rejected = run_script(model_dir, "extract_embeddings.py", "--checkpoint=" + tuned,
                          "--config=config.json", "--input-file=data/finetune_test.tfrecord",
                          "--split=test", "--output=project_results/rejected.npz", success=False)
    assert "verified label-independent" in rejected.stderr
    run_script(model_dir, "inspect_checkpoint.py", checkpoint)
    manifest_path = model_dir / "data/finetune_splits.json"
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest["splits"]["test"]["records"] += 1
    write_json(manifest_path, manifest)
    rejected = run_script(model_dir, "run_finetune.py", *args, "--do_train=False", "--load_cross=False",
                          "--init_checkpoint=" + frozen, success=False)
    assert "Incomplete record coverage" in rejected.stderr
    manifest_path.write_text(original)
    # Corrupt input must fail instead of saving an apparently successful checkpoint.
    (model_dir / "data/broken.tfrecord").write_bytes(b"broken")
    rejected = run_script(model_dir, "run_finetune.py", *args, "--split_manifest=",
                          "--init_checkpoint=" + checkpoint, "--train_input_file=data/broken.tfrecord",
                          "--checkpointDir=failed_run", success=False)
    assert not (model_dir / "failed_run/bert_finetune.index").exists()
