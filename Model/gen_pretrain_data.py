import os
import glob
import gzip
import struct
import numpy as np
import pickle as pkl
from tqdm import tqdm
import collections
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import time
import math
from vocab import FreqVocab
from research_protocol import sha256_files, write_json

tf.logging.set_verbosity(tf.logging.INFO)

rng = random.Random()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_seq_length", 100, "Must match downstream sequence length")
flags.DEFINE_float("masked_lm_prob", 0.15, "Fraction of transaction tokens selected for prediction")
flags.DEFINE_float("mask_prob", 0.8, "Selected-token probability of [MASK] replacement")
flags.DEFINE_float("random_token_prob", 0.1, "Selected-token probability of random replacement")
flags.DEFINE_bool("mask_self_token", False, "Include the account self token in the prediction objective")
flags.DEFINE_bool("do_eval", False,
                  "Diagnostic run only: hold out 20% of accounts from pretraining to measure masked-address "
                  "loss. The resulting encoder never trains on those accounts; use the default (False) "
                  "to pretrain on all accounts for downstream experiments.")
flags.DEFINE_bool("do_embed", False,
                  "Write an unmasked, label-free TFRecord for unsupervised embedding export")
flags.DEFINE_integer("dupe_factor", 1, "Number of independently masked copies per window")
flags.DEFINE_string("data_dir", './data/', "data dir.")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")
flags.DEFINE_bool("total_drop", False, "Use the same repeated-transaction preprocessing as downstream")
flags.DEFINE_integer("seed", 12345, "Masking, shuffle and window sampling seed")
flags.DEFINE_integer("records_per_shard", 25000,
                     "Maximum records in one TFRecord shard; avoids large-file corruption on Windows")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, address, tokens, masked_lm_positions, masked_lm_labels):

        self.address = [address]
        self.tokens = list(map(lambda x: x[0], tokens))
        self.block_timestamps = list(map(lambda x: x[2], tokens))
        self.values = list(map(lambda x: x[3], tokens))

        def map_io_flag(token):
            flag = token[4]
            if flag == "OUT":
                return 1
            elif flag == "IN":
                return 2
            else:
                return 0

        self.io_flags = list(map(map_io_flag, tokens))
        self.cnts = list(map(lambda x: x[5], tokens))
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def shard_paths(output_file, records, records_per_shard):
    if records_per_shard < 1:
        raise ValueError("records_per_shard must be positive")
    shard_count = max(1, math.ceil(records / records_per_shard))
    if shard_count == 1:
        return [output_file]
    stem, extension = os.path.splitext(output_file)
    return ["%s-%05d-of-%05d%s" % (stem, index, shard_count, extension)
            for index in range(shard_count)]


def _crc32c_table():
    table = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        table.append(crc & 0xffffffff)
    return table


CRC32C_TABLE = _crc32c_table()


def masked_crc32c(value):
    crc = 0xffffffff
    for byte in value:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xff] ^ (crc >> 8)
    crc = (~crc) & 0xffffffff
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xffffffff


def write_tfrecord_record(stream, serialized):
    length = struct.pack("<Q", len(serialized))
    stream.write(length)
    stream.write(struct.pack("<I", masked_crc32c(length)))
    stream.write(serialized)
    stream.write(struct.pack("<I", masked_crc32c(serialized)))


def remove_stale_shards(output_file, output_files):
    stem, extension = os.path.splitext(output_file)
    for stale in glob.glob(stem + "*" + extension):
        if stale not in output_files:
            os.remove(stale)


def gen_samples(sequences,
                dupe_factor,
                masked_lm_prob,
                max_predictions_per_seq,
                rng,
                vocab_words):
    instances = []
    for step in range(dupe_factor):
        start = time.time()
        for tokens in sequences:
            address, tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                tokens, masked_lm_prob, max_predictions_per_seq, rng,
                FLAGS.mask_prob, FLAGS.random_token_prob,
                FLAGS.mask_self_token, vocab_words=vocab_words)
            instances.append(TrainingInstance(
                address=address, tokens=tokens,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels))
        print("step=%d, time=%.2f" % (step, time.time() - start))
    print("=======Finish========")
    return instances


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, rng,
                                 mask_prob=0.8, random_token_prob=0.1,
                                 mask_self_token=False, vocab_words=None):
    """Creates the predictions for the masked LM objective."""

    address = tokens[0][0]
    if mask_prob < 0 or random_token_prob < 0 or mask_prob + random_token_prob > 1:
        raise ValueError("Replacement probabilities must be nonnegative and sum to at most one")
    cand_indexes = list(range(0 if mask_self_token else 1, len(tokens)))
    if not cand_indexes:
        raise ValueError("Sequence has no transaction token available for masking")

    rng.shuffle(cand_indexes)
    output_tokens = [list(i) for i in tokens]  # note that change the value of output_tokens will also change tokens
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(len(tokens) * masked_lm_prob)))
    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index][0]))
        replacement = rng.random()
        if replacement < mask_prob:
            output_tokens[index][0] = "[MASK]"
        elif replacement < mask_prob + random_token_prob:
            if not vocab_words:
                raise ValueError("Random replacement requires vocabulary tokens")
            output_tokens[index][0] = vocab_words[rng.randrange(len(vocab_words))]
        # Otherwise keep the original token, matching BERT's 80/10/10 policy.

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def create_embedding_predictions(tokens):
    """Creates the predictions for the masked LM objective."""
    address = tokens[0][0]
    output_tokens = tokens
    masked_lm_positions = []
    masked_lm_labels = []
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def gen_embedding_samples(sequences):
    instances = []
    # create train
    start = time.time()
    for tokens in sequences:
        (address, tokens, masked_lm_positions,
         masked_lm_labels) = create_embedding_predictions(tokens)
        instance = TrainingInstance(
            address=address,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    end = time.time()
    print("=======Finish========")
    print("cost time:%.2f" % (end - start))
    return instances


def convert_timestamp_to_position(block_timestamps):
    position = [0]
    if len(block_timestamps) <= 1:
        return position
    last_ts = block_timestamps[1]
    idx = 1
    for b_ts in block_timestamps[1:]:
        if b_ts != last_ts:
            last_ts = b_ts
            idx += 1
        position.append(idx)
    return position


def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    records_by_file = [0] * len(output_files)
    for output_file in output_files:
        writers.append(gzip.open(output_file, "wb"))

    writer_index = 0
    total_written = 0

    for inst_index in tqdm(range(len(instances))):
        instance = instances[inst_index]
        input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        address = vocab.convert_tokens_to_ids(instance.address)
        counts = instance.cnts
        block_timestamps = instance.block_timestamps
        values = instance.values
        io_flags = instance.io_flags
        positions = convert_timestamp_to_position(block_timestamps)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length
        assert len(counts) <= max_seq_length
        assert len(values) <= max_seq_length
        assert len(io_flags) <= max_seq_length
        assert len(positions) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        counts += [0] * (max_seq_length - len(counts))
        values += [0] * (max_seq_length - len(values))
        io_flags += [0] * (max_seq_length - len(io_flags))
        positions += [0] * (max_seq_length - len(positions))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(counts) == max_seq_length
        assert len(values) == max_seq_length
        assert len(io_flags) == max_seq_length
        assert len(positions) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        features["address"] = create_int_feature(address)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_positions"] = create_int_feature(positions)
        features["input_counts"] = create_int_feature(counts)
        features["input_io_flags"] = create_int_feature(io_flags)
        features["input_values"] = create_int_feature(values)

        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        write_tfrecord_record(writers[writer_index], tf_example.SerializeToString())
        records_by_file[writer_index] += 1
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(instance.tokens))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feature_name,
                                            " ".join([str(x)
                                                      for x in values])))

    for writer in writers:
        writer.close()

    for output_file, expected_records in zip(output_files, records_by_file):
        try:
            records = sum(1 for _ in tf.io.tf_record_iterator(output_file, options="GZIP"))
        except tf.errors.DataLossError as error:
            raise ValueError("Generated corrupt TFRecord: " + output_file) from error
        if records != expected_records:
            raise ValueError("Unexpected TFRecord record count: " + output_file)

    tf.logging.info("Wrote %d total instances", total_written)



def total_repeat_drop(eoa2seq):
    """
    totally drop the repeat transaction based on time.
    """
    new_eoa2seq = {}
    for eoa, seq in eoa2seq.items():
        new_seq = []
        exist_addr = set()
        for trans in seq:
            if trans[0] not in exist_addr:
                exist_addr.add(trans[0])
                new_seq.append(trans)

        new_eoa2seq[eoa] = new_seq

    return new_eoa2seq


def main():
    rng.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    max_predictions_per_seq = math.ceil(FLAGS.max_seq_length * FLAGS.masked_lm_prob)
    sliding_step = round(FLAGS.max_seq_length * 0.6)
    print("MAX_SEQUENCE_LENGTH:", FLAGS.max_seq_length)
    print("MAX_PREDICTIONS_PER_SEQ:", max_predictions_per_seq)
    print("SLIDING_STEP:", sliding_step)
    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open(os.path.join(FLAGS.data_dir, "eoa2seq.pkl"), "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))
    vocab.update(eoa2seq)
    # generate mapping
    vocab.generate_vocab()

    # save vocab
    print("token_size:{}".format(len(vocab.vocab_words)))
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pkl.dump(vocab, output_file, protocol=2)

    print("===========Original===========")
    length_list = []
    for eoa in eoa2seq.keys():
        seq = eoa2seq[eoa]
        length_list.append(len(seq))

    length_list = np.array(length_list)
    print("Median:", np.median(length_list))
    print("Mean:", np.mean(length_list))
    print("Seq num:", len(length_list))

    if FLAGS.total_drop:
        eoa2seq = total_repeat_drop(eoa2seq)

    print("==========After Reduce==========")
    length_list = []
    for eoa in eoa2seq.keys():
        seq = eoa2seq[eoa]
        length_list.append(len(seq))

    length_list = np.array(length_list)
    print("Median:", np.median(length_list))
    print("Mean:", np.mean(length_list))
    print("Seq num:", len(length_list))

    # clip
    max_num_tokens = FLAGS.max_seq_length - 1
    seqs = []
    idx = 0
    for eoa, seq in eoa2seq.items():
        if len(seq) <= max_num_tokens:
            seqs.append([[eoa, 0, 0, 0, 0, 0]])
            seqs[idx] += seq
            idx += 1
        elif len(seq) > max_num_tokens:
            beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * sliding_step))
            beg_idx.append(0)

            if len(beg_idx) > 500:
                beg_idx = list(np.random.permutation(beg_idx)[:500])
                for i in beg_idx:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

            else:
                for i in beg_idx[::-1]:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

    if FLAGS.do_embed:
        print("===========Generate Embedding Samples==========")
        write_instance = gen_embedding_samples(seqs)
        output_filename = FLAGS.data_dir + "embed.tfrecord"
        tf.logging.info("*** Writing to output embedding files ***")
        tf.logging.info("  %s", output_filename)

        embed_files = shard_paths(output_filename, len(write_instance), FLAGS.records_per_shard)
        remove_stale_shards(output_filename, embed_files)
        write_instance_to_example_files(write_instance, FLAGS.max_seq_length,
                                        max_predictions_per_seq, vocab, embed_files)

    rng.shuffle(seqs)

    eval_output_filename = FLAGS.data_dir + "test.tfrecord"
    if not FLAGS.do_eval:
        # Remove artifacts of an earlier diagnostic run so a stale holdout cannot be mistaken for this one.
        for stale in (eval_output_filename, FLAGS.data_dir + "pretrain_split.json"):
            if os.path.exists(stale):
                os.remove(stale)

    if FLAGS.do_eval:
        print("========Generate Diagnostic Holdout (excluded from pretraining)========")
        accounts = sorted(eoa2seq)
        rng.shuffle(accounts)
        eval_accounts = set(accounts[:max(1, round(len(accounts) * 0.2))])
        eval_seqs = [seq for seq in seqs if seq[0][0] in eval_accounts]
        seqs = [seq for seq in seqs if seq[0][0] not in eval_accounts]
        if not eval_seqs or not seqs:
            raise ValueError("Account-disjoint pretraining evaluation requires at least two accounts")

        eval_normal_instances = gen_samples(eval_seqs,
                                            dupe_factor=FLAGS.dupe_factor,
                                            masked_lm_prob=FLAGS.masked_lm_prob,
                                            max_predictions_per_seq=max_predictions_per_seq,
                                            rng=rng,
                                            vocab_words=vocab.vocab_words[3:])

        eval_write_instance = eval_normal_instances
        rng.shuffle(eval_write_instance)
        tf.logging.info("*** Writing to Testing files ***")
        tf.logging.info("  %s", eval_output_filename)

        eval_files = shard_paths(eval_output_filename, len(eval_write_instance), FLAGS.records_per_shard)
        remove_stale_shards(eval_output_filename, eval_files)
        write_instance_to_example_files(eval_write_instance, FLAGS.max_seq_length,
                                        max_predictions_per_seq, vocab, eval_files)
        import json
        with open(FLAGS.data_dir + "pretrain_split.json", "w") as output:
            json.dump({"seed": FLAGS.seed, "split_unit": "account",
                       "train_accounts": sorted(vocab.convert_tokens_to_ids(
                           [account for account in accounts if account not in eval_accounts])),
                       "validation_accounts": sorted(vocab.convert_tokens_to_ids(eval_accounts)),
                       "overlapping_accounts": 0}, output, indent=2)

    print("========Generate Training Samples========")
    normal_instances = gen_samples(seqs,
                                   dupe_factor=FLAGS.dupe_factor,
                                   masked_lm_prob=FLAGS.masked_lm_prob,
                                   max_predictions_per_seq=max_predictions_per_seq,
                                   rng=rng,
                                   vocab_words=vocab.vocab_words[3:])

    write_instance = normal_instances
    rng.shuffle(write_instance)

    output_filename = FLAGS.data_dir + "train.tfrecord"
    tf.logging.info("*** Writing to Training files ***")
    tf.logging.info("  %s", output_filename)

    train_files = shard_paths(output_filename, len(write_instance), FLAGS.records_per_shard)
    remove_stale_shards(output_filename, train_files)
    write_instance_to_example_files(write_instance, FLAGS.max_seq_length,
                                    max_predictions_per_seq, vocab, train_files)

    manifest = {
        "seed": FLAGS.seed, "max_seq_length": FLAGS.max_seq_length,
        "total_drop": FLAGS.total_drop, "masked_lm_probability": FLAGS.masked_lm_prob,
        "mask_probability": FLAGS.mask_prob, "random_token_probability": FLAGS.random_token_prob,
        "mask_self_token": FLAGS.mask_self_token, "dupe_factor": FLAGS.dupe_factor,
        "tfrecord_compression": "GZIP",
        "train_sha256": sha256_files(train_files), "train_files": train_files,
        "vocab_sha256": sha256_files([vocab_file_name]),
        "validation_sha256": sha256_files(eval_files) if FLAGS.do_eval else None,
        "account_disjoint_validation": FLAGS.do_eval,
        # A holdout run trains on a strict subset of accounts. Its checkpoints are diagnostics, not
        # the encoder for downstream experiments.
        "diagnostic_holdout": FLAGS.do_eval,
        "train_covers_all_accounts": not FLAGS.do_eval,
    }
    write_json(FLAGS.data_dir + "pretrain_manifest.json", manifest)

    return


if __name__ == '__main__':
    main()


