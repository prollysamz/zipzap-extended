from tqdm import tqdm
import pandas as pd
from research_protocol import split_accounts, write_json, sha256_file
import os
import collections
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pickle as pkl
import time

flags = tf.flags
FLAGS = flags.FLAGS

rng = random.Random()

## parameters
flags.DEFINE_integer("max_seq_length", 100, "max sequence length.")
flags.DEFINE_string("data_dir", './data/', "data dir.")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")
flags.DEFINE_bool("total_drop", False, "whether to drop")
flags.DEFINE_integer("seed", 12345, "Split and window sampling seed")

class FinetuneInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, address, tokens, label):

        self.address = [address]
        self.tokens = list(map(lambda x: x[0], tokens))
        self.block_timestamps = list(map(lambda x: x[2], tokens))
        self.values = list(map(lambda x: x[3], tokens))
        self.label = label

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

def gen_finetune_samples(sequences, label_list):
    instances = []
    # create train
    start = time.time()
    for i in tqdm(range(len(sequences))):

        tokens = sequences[i]
        address = tokens[0][0]
        instance = FinetuneInstance(
            address=address,
            tokens=tokens,
            label=label_list[i])
        instances.append(instance)

    end = time.time()
    print("=======Finish========")
    print("cost time:%.2f" % (end - start))
    return instances

def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature

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

def write_finetune_instance_to_example_files(instances, max_seq_length, vocab, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    records_by_file = [0] * len(output_files)
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0

    for inst_index in tqdm(range(len(instances))):
        instance = instances[inst_index]
        input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        address = vocab.convert_tokens_to_ids(instance.address)
        counts = list(instance.cnts)
        block_timestamps = instance.block_timestamps
        values = list(instance.values)
        io_flags = list(instance.io_flags)
        positions = convert_timestamp_to_position(block_timestamps)
        label = [instance.label]

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

        features = collections.OrderedDict()
        features["address"] = create_int_feature(address)
        features["label"] = create_float_feature(label)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_positions"] = create_int_feature(positions)
        features["input_counts"] = create_int_feature(counts)
        features["input_io_flags"] = create_int_feature(io_flags)
        features["input_values"] = create_int_feature(values)
        features["input_mask"] = create_int_feature(input_mask)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
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
            records = sum(1 for _ in tf.io.tf_record_iterator(output_file))
        except tf.errors.DataLossError as error:
            raise ValueError("Generated corrupt TFRecord: " + output_file) from error
        if records != expected_records:
            raise ValueError("Unexpected TFRecord record count: " + output_file)

    tf.logging.info("Wrote %d total instances", total_written)


def total_repeat_drop(eoa2seq):
    """
    totally drop the repeat part.
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


if __name__ == '__main__':

    rng.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    sliding_step = round(FLAGS.max_seq_length * 0.6)
    print("MAX_SEQUENCE_LENGTH:", FLAGS.max_seq_length)
    print("SLIDING_STEP:", sliding_step)

    # load label
    phisher_account = pd.read_csv("../Data/phisher_account.txt", names=["account"])
    phisher_account_set = set(phisher_account.account.values)

    # load vocab
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename
    with open(vocab_file_name, "rb") as f:
        vocab = pkl.load(f)

    source_file = os.path.join(FLAGS.data_dir, "eoa2seq.pkl")
    with open(source_file, "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))

    if FLAGS.total_drop:
        eoa2seq = total_repeat_drop(eoa2seq)

    splits = split_accounts(eoa2seq.keys(),
                            {a: int(a in phisher_account_set) for a in eoa2seq}, FLAGS.seed)
    account_split = {account: split for split, accounts in splits.items() for account in accounts}

    label_list = []
    # clip and add label
    def is_phish(address):
        if address in phisher_account_set:
            return 1.0
        else:
            return 0.0

    max_num_tokens = FLAGS.max_seq_length - 1
    seqs = []
    idx = 0
    for eoa, seq in eoa2seq.items():
        if len(seq) <= max_num_tokens:
            seqs.append([[eoa, 0, 0, 0, 0, 0]])
            seqs[idx] += seq
            idx += 1
            label_list.append(is_phish(eoa))

        elif len(seq) > max_num_tokens:
            beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * sliding_step))
            beg_idx.append(0)

            if len(beg_idx) > 500:
                beg_idx = list(np.random.permutation(beg_idx)[:500])
                for i in beg_idx:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1
                    label_list.append(is_phish(eoa))

            else:
                for i in beg_idx[::-1]:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1
                    label_list.append(is_phish(eoa))

    grouped = {name: ([], []) for name in splits}
    for seq, label in zip(seqs, label_list):
        sequences, labels = grouped[account_split[seq[0][0]]]
        sequences.append(seq)
        labels.append(label)
    manifest = {"seed": FLAGS.seed, "evaluation_setting": "transductive",
                "split_label_use": "stratified_using_known_phishing_labels",
                "negative_class": "accounts_without_known_phishing_labels",
                "vocab_sha256": sha256_file(vocab_file_name),
                "source_sha256": sha256_file(source_file),
                "known_phishing_sha256": sha256_file("../Data/phisher_account.txt"),
                "max_seq_length": FLAGS.max_seq_length, "total_drop": FLAGS.total_drop,
                "amount_feature": "transaction_amount_bucket", "splits": {}}
    for name, (sequences, labels) in grouped.items():
        instances = gen_finetune_samples(sequences, labels)
        rng.shuffle(instances)
        output_filename = os.path.join(FLAGS.data_dir, "finetune_" + name + ".tfrecord")
        write_finetune_instance_to_example_files(instances, FLAGS.max_seq_length, vocab, [output_filename])
        manifest["splits"][name] = {
            "account_ids": sorted(vocab.convert_tokens_to_ids(sorted(splits[name]))),
            "records": len(instances), "sha256": sha256_file(output_filename)}
    write_json(os.path.join(FLAGS.data_dir, "finetune_splits.json"), manifest)
    print("Finished..")
