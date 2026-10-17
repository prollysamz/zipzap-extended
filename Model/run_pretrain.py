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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import modeling
sys.path.append("..")
import optimization
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import pickle as pkl
# import time
from timeit import default_timer as timer
import math
import json
from research_protocol import save_provenance, sha256_file, read_provenance

flags = tf.flags
FLAGS = flags.FLAGS

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## Required parameters
flags.DEFINE_string(
    "bert_config_file", "./zipzap_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", "./data/train.tfrecord",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", "./data/test.tfrecord",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "checkpointDir", "ckpt_dir",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_integer("max_seq_length", 100, "Must match downstream sequence length")
flags.DEFINE_float("masked_lm_prob", 0.15, "Fraction of transaction tokens selected for prediction")
flags.DEFINE_bool("do_train", True, "")
flags.DEFINE_bool("do_eval", False, "")
flags.DEFINE_integer("batch_size", 256, "")
flags.DEFINE_integer("epoch", 1, "Number of passes over generated records")
flags.DEFINE_float("learning_rate", 1e-4, "")
flags.DEFINE_integer("num_train_steps", 10000000, "Number of training steps.")
flags.DEFINE_integer("num_warmup_steps", 100, "Number of warmup steps.")
flags.DEFINE_integer("save_checkpoints_steps", 8000, "")
flags.DEFINE_integer("neg_sample_num", 512, "The number of negative samples in a batch")
flags.DEFINE_string("neg_strategy", "zip", "Strategy of negative sampling")
flags.DEFINE_string("data_dir", './data/', "data dir.")
flags.DEFINE_bool("cross_share", True, "whether to share or not")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")
flags.DEFINE_string("pretrain_manifest", "./data/pretrain_manifest.json", "Pretraining data contract")
flags.DEFINE_integer("seed", 12345, "NumPy, TensorFlow and shuffle seed")

MAX_PREDICTIONS_PER_SEQ = math.ceil(FLAGS.max_seq_length * FLAGS.masked_lm_prob)

print("MAX_SEQUENCE_LENGTH:", FLAGS.max_seq_length)
print("MAX_PREDICTIONS_PER_SEQ:", MAX_PREDICTIONS_PER_SEQ)


def input_fn(input_files,
             is_training,
             num_cpu_threads=4):
    """ The actual input function"""

    name_to_features = {
        "address":
            tf.FixedLenFeature([1], tf.int64),
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
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([MAX_PREDICTIONS_PER_SEQ], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([MAX_PREDICTIONS_PER_SEQ], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([MAX_PREDICTIONS_PER_SEQ], tf.float32)
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
             num_train_steps, num_warmup_steps, use_one_hot_embeddings):
    """The `model_fn` for TPUEstimator."""

    # tf.logging.info("*** Features ***")
    # for name in sorted(features.keys()):
    #     tf.logging.info("name = %s, shape = %s" % (name,
    #                                                features[name].shape))

    input_ids = features["input_ids"]
    input_positions = features["input_positions"]
    input_mask = features["input_mask"]
    input_io_flags = features["input_io_flags"]
    input_values = features["input_values"]
    input_counts = features["input_counts"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]

    is_training = (mode == "train")

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

    embedding_table_list, factorize_table_list = model.get_embedding_table()

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output_negative_sampling(
        bert_config,
        model.get_sequence_output(),
        embedding_table_list,
        factorize_table_list,
        masked_lm_positions,
        masked_lm_ids,
        masked_lm_weights,
        vocab)  # model use the token embedding table as the output_weights

    total_loss = masked_lm_loss
    tvars = tf.trainable_variables()
    initialized_variable_names = {}

    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        missing_encoder = [var.name for var in tvars
                           if var.name.startswith("bert/") and var.name not in initialized_variable_names]
        if missing_encoder:
            raise ValueError("Checkpoint did not initialize encoder variables: " + str(missing_encoder[:5]))
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

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
                                                 num_warmup_steps, False)

        return model, train_op, total_loss
        # output_spec = tf.estimator.EstimatorSpec(
        #     mode=mode,
        #     loss=total_loss,
        #     train_op=train_op,
        #     scaffold=scaffold_fn)

    elif mode == "eval":
        # Sampled candidates change each step, so column-zero top-1 is not
        # full-vocabulary accuracy. Report only corrected sampled loss.
        return model, total_loss

    else:
        raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))


def get_masked_lm_output_negative_sampling(bert_config, input_tensor,
                                           embedding_table_list, factorize_table_list,
                                           positions, label_ids, label_weights, vocab):
    """Get loss and log probs for the masked LM."""

    # Normal address IDs begin at 4. Candidate samplers use zero-based class
    # IDs, so convert the actual targets before sampling and map candidates
    # back to vocabulary IDs for the factorized embedding lookup.
    word_num = len(vocab.vocab_words) - 3
    flat_label_ids = tf.reshape(label_ids, [-1])
    true_classes = tf.cast(tf.maximum(flat_label_ids - 4, 0)[:, None], tf.int64)
    sample_count = min(FLAGS.neg_sample_num, word_num - 1)
    if sample_count < 1:
        raise ValueError("Negative sampling requires at least two address tokens")

    if FLAGS.neg_strategy == "uniform":
        sampled_classes, true_expected_count, sampled_expected_count = tf.nn.uniform_candidate_sampler(
                                                        true_classes=true_classes,
                                                        num_true=1,
                                                        num_sampled=sample_count,
                                                        unique=True,
                                                        range_max=word_num)

    elif FLAGS.neg_strategy == "zip":
        sampled_classes, true_expected_count, sampled_expected_count = tf.nn.log_uniform_candidate_sampler(
                                                            true_classes=true_classes,
                                                            num_true=1,
                                                            num_sampled=sample_count,
                                                            unique=True,
                                                            range_max=word_num)

    elif FLAGS.neg_strategy == "freq":
        # negative sample based on frequency
        sampled_classes, true_expected_count, sampled_expected_count = tf.nn.fixed_unigram_candidate_sampler(
                                                              true_classes=true_classes,
                                                              num_true=1,
                                                              num_sampled=sample_count,
                                                              unique=True,
                                                              range_max=word_num,
                                                              unigrams=list(
                                                                  map(lambda x: pow(x, 1 / 1), vocab.frequency[3:]))
                                                              )

    else:
        raise ValueError("Please select correct negative sampling strategy: uniform, zip, .")

    sampled_class_ids = sampled_classes
    neg_ids = tf.cast(sampled_class_ids, tf.int32) + 4

    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        label_weights = tf.reshape(label_weights, [-1])

        pos_output_weights = modeling.new_embedding_lookup(label_ids,
                                                           bert_config.bucket_list,
                                                           embedding_table_list,
                                                           factorize_table_list)

        pos_output_weights = tf.reshape(pos_output_weights, [-1, pos_output_weights.shape[-1]])

        neg_ids = tf.expand_dims(neg_ids, axis=0)
        neg_output_weights = modeling.new_embedding_lookup(neg_ids,
                                                           bert_config.bucket_list,
                                                           embedding_table_list,
                                                           factorize_table_list)

        neg_output_weights = tf.reshape(neg_output_weights, [-1, neg_output_weights.shape[-1]])

        pos_logits = tf.reduce_sum(tf.multiply(input_tensor, pos_output_weights), axis=-1)
        # Correct for the non-uniform proposal distribution. Without this,
        # frequent addresses are systematically penalized.
        pos_logits -= tf.math.log(tf.maximum(tf.reshape(true_expected_count, [-1]), 1e-10))
        pos_logits = tf.expand_dims(pos_logits, axis=1)
        neg_logits = tf.matmul(input_tensor, neg_output_weights, transpose_b=True)
        neg_logits -= tf.math.log(tf.maximum(sampled_expected_count[None, :], 1e-10))

        # A sampled candidate equal to the row's true token is not a negative.
        accidental_hits = tf.equal(true_classes, sampled_class_ids[None, :])
        neg_logits = tf.where(accidental_hits,
                              tf.fill(tf.shape(neg_logits), tf.constant(-1e9, tf.float32)),
                              neg_logits)

        logits = tf.concat([pos_logits, neg_logits], axis=1)
        log_probs = tf.nn.log_softmax(logits, -1)
        per_example_loss = -log_probs[:, 0]
        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]
    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


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


def main(_):
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    if FLAGS.do_train:
        mode = "train"
        input_files = FLAGS.train_input_file
        # load data
        features = input_fn(input_files, is_training=True)

    elif FLAGS.do_eval:
        mode = "eval"
        input_files = FLAGS.test_input_file
        features = input_fn(input_files, is_training=False)

    else:
        raise ValueError("Only TRAIN and EVAL modes are supported.")

    # modeling
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    # load vocab
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename
    with open(vocab_file_name, "rb") as f:
        vocab = pkl.load(f)
    with open(FLAGS.pretrain_manifest, encoding="utf-8") as source:
        data_manifest = json.load(source)
    expected_hash = data_manifest["train_sha256"] if FLAGS.do_train else data_manifest["validation_sha256"]
    if expected_hash is None:
        raise ValueError("No diagnostic holdout exists in this pretraining data. "
                         "Regenerate it with gen_pretrain_data.py --do_eval=True (diagnostic runs only).")
    if expected_hash != sha256_file(input_files):
        raise ValueError("Pretraining TFRecord does not match its manifest")
    if data_manifest["max_seq_length"] != FLAGS.max_seq_length:
        raise ValueError("Pretraining sequence length differs from its data manifest")
    if max(vocab.id_to_tokens) >= bert_config.vocab_size:
        raise ValueError("Vocabulary IDs exceed configured embedding coverage")
    if FLAGS.init_checkpoint:
        parent = read_provenance(FLAGS.init_checkpoint)
        if parent.get("verified") and (parent["config_sha256"] != sha256_file(FLAGS.bert_config_file)
                                      or parent["vocab_sha256"] != sha256_file(vocab_file_name)):
            raise ValueError("Initial checkpoint config or vocabulary mismatch")

    def save_checkpoint(sess, saver, step):
        if step <= 0:
            raise ValueError("Pretraining input contained no records")
        prefix = saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(step)))
        save_provenance(prefix, "masked_address_prediction", FLAGS.bert_config_file,
                        vocab_file_name, parent=FLAGS.init_checkpoint,
                        seed=FLAGS.seed, steps=step, cross_share=FLAGS.cross_share,
                        evaluation_setting="transductive",
                        diagnostic_holdout=data_manifest.get("diagnostic_holdout", False),
                        trained_on_all_accounts=data_manifest.get("train_covers_all_accounts", False),
                        max_seq_length=data_manifest["max_seq_length"],
                        total_drop=data_manifest["total_drop"],
                        masked_lm_probability=data_manifest["masked_lm_probability"],
                        mask_probability=data_manifest["mask_probability"],
                        random_token_probability=data_manifest["random_token_probability"],
                        mask_self_token=data_manifest["mask_self_token"],
                        negative_sampling_strategy=FLAGS.neg_strategy,
                        negative_sample_count=min(FLAGS.neg_sample_num, len(vocab.vocab_words) - 4),
                        sampled_softmax_log_q_correction=True,
                        accidental_hits_removed=True,
                        position_specific_output_bias=False,
                        training_input_sha256=sha256_file(FLAGS.train_input_file))

    if FLAGS.do_train:
        bert_model, train_op, total_loss = model_fn(features, mode, bert_config, vocab, FLAGS.init_checkpoint,
                                                    FLAGS.learning_rate,
                                                    FLAGS.num_train_steps, FLAGS.num_warmup_steps, False)
        # saver define
        tvars = tf.trainable_variables()
        saver = tf.train.Saver(max_to_keep=30, var_list=tvars)

        # start session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            losses = []
            iter = 0
            # start = time.time()
            start = timer()
            while True:
                try:
                    _, loss = sess.run([train_op, total_loss])
                    if not np.isfinite(loss):
                        raise ValueError("Non-finite pretraining loss")
                    # loss = sess.run([total_loss])
                    losses.append(loss)

                    if iter % 500 == 0:
                        # end = time.time()
                        end = timer()
                        loss = np.mean(losses)
                        print("iter=%d, loss=%f, time=%.3fs" % (iter, loss, end - start))
                        losses = []
                        # start = time.time()
                        start = timer()

                    if iter % FLAGS.save_checkpoints_steps == 0 and iter > 0:
                        save_checkpoint(sess, saver, iter + 1)

                    iter += 1

                except tf.errors.OutOfRangeError:
                    print("Out of Sequence, end of training...")
                    # save model
                    save_checkpoint(sess, saver, iter)
                    break
                except Exception as e:
                    print("Unexpected error:", e)
                    raise

    elif FLAGS.do_eval:
        # must have checkpoint
        if FLAGS.init_checkpoint == None:
            raise ValueError("Must need a checkpoint for evaluation")

        bert_model, total_loss = model_fn(features, mode, bert_config, vocab, FLAGS.init_checkpoint,
                                          FLAGS.learning_rate,
                                          FLAGS.num_train_steps, FLAGS.num_warmup_steps, False)

        # start session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            losses = []
            iter = 0
            # start = time.time()
            start = timer()
            while True:
                try:
                    loss = sess.run(total_loss)
                    losses.append(loss)

                    if iter % 500 == 0:
                        # end = time.time()
                        end = timer()
                        print("iter=%d, time=%.3fs" % (iter, end - start))
                        # start = time.time()
                        start = timer()
                    iter += 1

                except tf.errors.OutOfRangeError:
                    print("Out of Sequence")
                    # save model
                    # saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(iter)))
                    break

            if not losses:
                raise ValueError("Evaluation input contained no records")
            final_loss = np.mean(losses)
            eval_sample_num = len(losses)

            print("========Evaluation Results==========")
            print("sample_num=%d, loss=%.2f" % (eval_sample_num, final_loss))

    else:
        raise ValueError("Only TRAIN and EVAL modes are supported.")

    return


if __name__ == '__main__':
    tf.app.run()
