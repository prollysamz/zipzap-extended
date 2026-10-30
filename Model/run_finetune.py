# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append("..")
import optimization
import collections
import re
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import modeling
import numpy as np
import pickle as pkl
import time
import json
from evaluation import aggregate_accounts, evaluate, select_f1_threshold
from research_protocol import (read_provenance, save_provenance, sha256_file,
                               validate_split_accounts, write_json)

flags = tf.flags
FLAGS = flags.FLAGS

def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t
    return example

def input_fn(input_files,
             is_training,
             num_cpu_threads=4):
    """ The actual input function"""

    name_to_features = {
        "address":
            tf.FixedLenFeature([1], tf.int64),
        "label":
            tf.FixedLenFeature([1], tf.float32),
        "input_ids":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_positions":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_counts":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_io_flags":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_values":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64)
    }

    if is_training:
        d = tf.data.TFRecordDataset(input_files)
        d = d.repeat(FLAGS.epoch).shuffle(100, seed=FLAGS.seed)

    else:
        d = tf.data.TFRecordDataset(input_files)

    d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=num_cpu_threads)
    d = d.batch(batch_size=FLAGS.batch_size)

    iterator = d.make_one_shot_iterator()
    features = iterator.get_next()

    return features


def model_fn(features, mode, bert_config, vocab, init_checkpoint, learning_rate,
             num_train_steps, num_warmup_steps, load_cross, use_one_hot_embeddings):
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.info("name = %s, shape = %s" % (name,
                                                   features[name].shape))

    label = tf.reshape(features["label"], [-1])
    input_ids = features["input_ids"]
    input_positions = features["input_positions"]
    input_mask = features["input_mask"]
    input_io_flags = features["input_io_flags"]
    input_values = features["input_values"]
    input_counts = features["input_counts"]

    is_training = (mode == "train" and not FLAGS.freeze_encoder)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_positions=input_positions,
        input_io_flags=input_io_flags,
        input_amounts=input_values,
        input_counts=input_counts,
        input_mask=input_mask,
        token_type_ids=None,
        use_one_hot_embeddings=use_one_hot_embeddings,
        cross_share=FLAGS.cross_share)

    transformer_output = model.get_sequence_output()
    print(transformer_output)
    with tf.variable_scope("MLP", reuse=tf.AUTO_REUSE):

        # inp = tf.reduce_mean(transformer_output, 1)
        inp = transformer_output[:,0,:]
        if FLAGS.freeze_encoder:
            inp = tf.stop_gradient(inp)

        dnn1 = tf.layers.dense(inp, FLAGS.hidden_size, activation=tf.nn.relu, name='f1')
        dnn2 = tf.layers.dense(dnn1, FLAGS.hidden_size, activation=tf.nn.relu, name='f2')
        logit = tf.reshape(tf.layers.dense(dnn2 + dnn1, 1, activation=None, name='logit'), [-1])
        y_hat = tf.sigmoid(logit)

    # print("--------------------")
    # print("label:", label)
    # print("logit:", logit)
    # print("--------------------")
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

    total_loss = loss
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        if load_cross:
            # multi-layer parameter loading

            name_to_variable = collections.OrderedDict()
            for var in tvars:
                name = var.name
                m = re.match("^(.*):\\d+$", name)
                if m is not None:
                    name = m.group(1)
                name_to_variable[name] = var

            for layer_index in range(bert_config.num_hidden_layers):

                assignment_map = collections.OrderedDict()
                for name in name_to_variable.keys():
                    if "layer_" + str(layer_index) in name:

                        var_name_list = name.split("/")
                        var_name_list[2] = "shared_layer"
                        load_name = "/".join(var_name_list)
                        # assignment_map[name] = new_name
                        assignment_map[load_name] = name
                        initialized_variable_names[name] = 1
                        initialized_variable_names[name + ":0"] = 1

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        missing_encoder = [var.name for var in tvars
                           if var.name.startswith("bert/") and var.name not in initialized_variable_names]
        if missing_encoder:
            raise ValueError("Checkpoint did not initialize encoder variables: " + str(missing_encoder[:5]))

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    if mode == "train":
        train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                 num_train_steps,
                                                 num_warmup_steps, False,
                                                 var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="MLP/")
                                                 if FLAGS.freeze_encoder else None)

        return model, train_op, total_loss

    elif mode == "eval":

        return model, y_hat, total_loss

    else:
        raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))


def load_split_manifest(vocab_file):
    if not FLAGS.split_manifest:
        return None
    with open(FLAGS.split_manifest, encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest["vocab_sha256"] != sha256_file(vocab_file):
        raise ValueError("Split manifest uses a different vocabulary")
    if manifest["max_seq_length"] != FLAGS.max_seq_length:
        raise ValueError("Split manifest uses a different sequence length")
    validate_split_accounts(manifest)
    return manifest


def check_input_file(manifest, split, path):
    if manifest and manifest["splits"][split]["sha256"] != sha256_file(path):
        raise ValueError("TFRecord does not match split manifest: " + split)


def score_checkpoint(checkpoint, input_file, split, config, vocab, manifest):
    check_input_file(manifest, split, input_file)
    addresses, labels, scores = [], [], []
    start = time.perf_counter()
    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.seed)
        features = input_fn(input_file, is_training=False)
        _, probabilities, _ = model_fn(
            features, "eval", config, vocab, None, FLAGS.learning_rate,
            FLAGS.num_train_steps, FLAGS.num_warmup_steps, False, False)
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            while True:
                try:
                    batch_ids, batch_labels, batch_scores = sess.run(
                        [features["address"], features["label"], probabilities])
                except tf.errors.OutOfRangeError:
                    break
                addresses.extend(np.asarray(batch_ids).reshape(-1).tolist())
                labels.extend(np.asarray(batch_labels).reshape(-1).tolist())
                scores.extend(np.asarray(batch_scores).reshape(-1).tolist())
    if manifest and len(addresses) != manifest["splits"][split]["records"]:
        raise ValueError("Incomplete record coverage for " + split)
    result = aggregate_accounts(addresses, labels, scores)
    if manifest and set(result[0]) != set(manifest["splits"][split]["account_ids"]):
        raise ValueError("Incomplete account coverage for " + split)
    return result, time.perf_counter() - start


def train_checkpoint(config, vocab, vocab_file, manifest):
    check_input_file(manifest, "train", FLAGS.train_input_file)
    if FLAGS.init_checkpoint:
        parent = read_provenance(FLAGS.init_checkpoint)
        if parent.get("verified") and (parent["config_sha256"] != sha256_file(FLAGS.bert_config_file)
                                      or parent["vocab_sha256"] != sha256_file(vocab_file)):
            raise ValueError("Initial checkpoint config or vocabulary mismatch")
        if parent.get("verified") and parent.get("objective") == "masked_address_prediction":
            if parent.get("max_seq_length") != FLAGS.max_seq_length:
                raise ValueError("Pretraining and downstream sequence lengths differ")
            if manifest and parent.get("total_drop") != manifest.get("total_drop"):
                raise ValueError("Pretraining and downstream repeat-dropping settings differ")
    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.seed)
        features = input_fn(FLAGS.train_input_file, is_training=True)
        _, train_op, loss = model_fn(
            features, "train", config, vocab, FLAGS.init_checkpoint, FLAGS.learning_rate,
            FLAGS.num_train_steps, FLAGS.num_warmup_steps, FLAGS.load_cross, False)
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            steps = 0
            while True:
                try:
                    _, value = sess.run([train_op, loss])
                except tf.errors.OutOfRangeError:
                    break
                if not np.isfinite(value):
                    raise ValueError("Non-finite training loss")
                steps += 1
                if steps % 100 == 0:
                    print("step=%d loss=%f" % (steps, value))
            if not steps:
                raise ValueError("Training input contained no records")
            checkpoint = os.path.join(FLAGS.checkpointDir,
                                      "bert_frozen_mlp" if FLAGS.freeze_encoder else "bert_finetune")
            saver.save(sess, checkpoint)
    save_provenance(checkpoint, "supervised", FLAGS.bert_config_file, vocab_file,
                    parent=FLAGS.init_checkpoint, frozen_encoder=FLAGS.freeze_encoder,
                    encoder_initialization="checkpoint" if FLAGS.init_checkpoint else "random",
                    steps=steps, seed=FLAGS.seed, cross_share=FLAGS.cross_share,
                    evaluation_setting="transductive",
                    split_manifest_sha256=sha256_file(FLAGS.split_manifest) if manifest else None)
    return checkpoint


def main(_):
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("Enable training or evaluation")
    if FLAGS.init_checkpoint and tf.gfile.IsDirectory(FLAGS.init_checkpoint):
        directory = FLAGS.init_checkpoint
        FLAGS.init_checkpoint = tf.train.latest_checkpoint(directory)
        if not FLAGS.init_checkpoint:
            raise ValueError("No checkpoint found in " + directory)
    np.random.seed(FLAGS.seed)
    config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    vocab_file = os.path.join(FLAGS.data_dir, FLAGS.vocab_filename)
    with open(vocab_file, "rb") as source:
        vocab = pkl.load(source)
    if max(vocab.id_to_tokens) >= config.vocab_size:
        raise ValueError("Vocabulary IDs exceed configured embedding coverage")
    manifest = load_split_manifest(vocab_file)
    tf.gfile.MakeDirs(FLAGS.checkpointDir)
    start = time.perf_counter()
    checkpoint = train_checkpoint(config, vocab, vocab_file, manifest) if FLAGS.do_train else FLAGS.init_checkpoint
    training_seconds = time.perf_counter() - start if FLAGS.do_train else None
    if FLAGS.do_eval:
        if not checkpoint:
            raise ValueError("Evaluation requires --init_checkpoint")
        provenance = read_provenance(checkpoint)
        if provenance.get("verified"):
            if provenance["config_sha256"] != sha256_file(FLAGS.bert_config_file):
                raise ValueError("Checkpoint config differs from the supplied config")
            if provenance["vocab_sha256"] != sha256_file(vocab_file):
                raise ValueError("Checkpoint vocabulary differs from the supplied vocabulary")
            if provenance["cross_share"] != FLAGS.cross_share:
                raise ValueError("Checkpoint uses a different cross_share setting")
        if FLAGS.threshold_protocol == "validation_f1":
            validation, _ = score_checkpoint(checkpoint, FLAGS.validation_input_file, "validation", config, vocab, manifest)
            threshold = select_f1_threshold(validation[1], validation[2])
        elif FLAGS.threshold_protocol == "fixed":
            threshold = FLAGS.fixed_threshold
            validation = None
        else:
            raise ValueError("Unknown threshold protocol")
        # Test data is read only after the threshold is fixed.
        test, inference_seconds = score_checkpoint(checkpoint, FLAGS.test_input_file, "test", config, vocab, manifest)
        if validation is not None and set(validation[0]).intersection(test[0]):
            raise ValueError("Validation and test accounts overlap")
        report = evaluate(test[1], test[2], threshold, FLAGS.alert_fraction)
        report.update(checkpoint=checkpoint, seed=FLAGS.seed,
                      threshold_protocol=FLAGS.threshold_protocol,
                      threshold_label_use=FLAGS.threshold_protocol == "validation_f1",
                      checkpoint_provenance=provenance, training_seconds=training_seconds,
                      inference_seconds=inference_seconds,
                      evaluation_setting="transductive", coverage_verified=manifest is not None)
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        write_json(os.path.join(FLAGS.output_dir, "metrics.json"), report)
        np.savez_compressed(os.path.join(FLAGS.output_dir, "test_scores.npz"),
                            account_ids=test[0], labels=test[1], scores=test[2])
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    flags.DEFINE_string("bert_config_file", "./zipzap_config.json", "Model configuration")
    flags.DEFINE_string("checkpointDir", "ckpt_dir", "Checkpoint output directory")
    flags.DEFINE_string("data_dir", "./data/", "Data directory")
    flags.DEFINE_string("vocab_filename", "vocab", "Vocabulary filename")
    flags.DEFINE_integer("batch_size", 256, "Batch size")
    flags.DEFINE_integer("num_train_steps", 10000000, "Learning rate decay steps")
    flags.DEFINE_integer("num_warmup_steps", 100, "Learning rate warmup steps")
    flags.DEFINE_integer("seed", 12345, "NumPy, TensorFlow and shuffle seed")
    flags.DEFINE_bool("do_train", True, "Whether to run training.")
    flags.DEFINE_bool("do_eval", True, "Whether to run evaluation.")
    flags.DEFINE_bool("cross_share", False, "whether to share or not")
    flags.DEFINE_bool("load_cross", True, "whether to load from cross")
    flags.DEFINE_integer("epoch", 1, "Epoch for finetune")
    flags.DEFINE_integer("max_seq_length", 100, "")
    flags.DEFINE_string("train_input_file", "./data/finetune_train.tfrecord", "Input train file for finetuning")
    flags.DEFINE_string("test_input_file", "./data/finetune_test.tfrecord", "Input test file for finetuning")
    flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
    flags.DEFINE_integer("hidden_size", 128, "Hidden size for downside MLP.")
    flags.DEFINE_float("learning_rate", 3e-4, "")
    flags.DEFINE_bool("freeze_encoder", False, "Train only the MLP on deterministic frozen representations")
    flags.DEFINE_string("validation_input_file", "./data/finetune_validation.tfrecord", "Validation TFRecord")
    flags.DEFINE_string("split_manifest", "./data/finetune_splits.json", "Coverage manifest; empty only for legacy data")
    flags.DEFINE_string("threshold_protocol", "validation_f1", "validation_f1 or fixed")
    flags.DEFINE_float("fixed_threshold", 0.5, "Predeclared threshold for fixed protocol")
    flags.DEFINE_float("alert_fraction", 0.01, "Fraction of accounts inspected for ranking metrics")
    flags.DEFINE_string("output_dir", "./project_results/supervised", "Evaluation artifacts")

    print("==========Parameters===========")
    print("cross_share:", FLAGS.cross_share)
    print("load_cross:", FLAGS.load_cross)
    print("learning_rate:", FLAGS.learning_rate)
    tf.app.run()
