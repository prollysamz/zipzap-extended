"""Export the 64-D pre-MLP ZipZap representation, averaged per account."""

import argparse
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow.compat.v1 as tf

import modeling

tf.disable_v2_behavior()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--bert_config_file", default="zipzap_config.json")
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cross_share", action="store_true")
    parser.add_argument("--include_classifier", action="store_true")
    return parser.parse_args()


def decode(record, length):
    spec = {
        "address": tf.FixedLenFeature([1], tf.int64),
        "label": tf.FixedLenFeature([1], tf.float32),
        "input_ids": tf.FixedLenFeature([length], tf.int64),
        "input_positions": tf.FixedLenFeature([length], tf.int64),
        "input_counts": tf.FixedLenFeature([length], tf.int64),
        "input_mask": tf.FixedLenFeature([length], tf.int64),
        "input_io_flags": tf.FixedLenFeature([length], tf.int64),
        "input_values": tf.FixedLenFeature([length], tf.int64),
    }
    item = tf.parse_single_example(record, spec)
    for key, value in list(item.items()):
        if value.dtype == tf.int64:
            item[key] = tf.to_int32(value)
    return item


def main():
    args = arguments()
    if not tf.gfile.Exists(args.input_file):
        raise FileNotFoundError(args.input_file)
    if not tf.gfile.Exists(args.checkpoint + ".index"):
        raise FileNotFoundError(args.checkpoint + ".index")

    dataset = tf.data.TFRecordDataset([args.input_file])
    dataset = dataset.map(lambda row: decode(row, args.max_seq_length), 4)
    features = dataset.batch(args.batch_size).make_one_shot_iterator().get_next()
    config = modeling.BertConfig.from_json_file(args.bert_config_file)
    model = modeling.BertModel(
        config=config,
        is_training=False,
        input_ids=features["input_ids"],
        input_positions=features["input_positions"],
        input_io_flags=features["input_io_flags"],
        input_amounts=features["input_values"],
        input_counts=features["input_counts"],
        input_mask=features["input_mask"],
        token_type_ids=None,
        use_one_hot_embeddings=False,
        cross_share=args.cross_share,
    )
    representation = model.get_sequence_output()[:, 0, :]
    classifier_score = None
    if args.include_classifier:
        with tf.variable_scope("MLP", reuse=tf.AUTO_REUSE):
            dense_1 = tf.layers.dense(
                representation, 128, activation=tf.nn.relu, name="f1")
            dense_2 = tf.layers.dense(
                dense_1, 128, activation=tf.nn.relu, name="f2")
            classifier_score = tf.sigmoid(tf.squeeze(tf.layers.dense(
                dense_2 + dense_1, 1, activation=None, name="logit")))
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    sums, counts, labels, score_sums = {}, {}, {}, {}
    window_count = 0

    with tf.Session(config=session_config) as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, args.checkpoint)
        while True:
            try:
                fetches = [features["address"], features["label"], representation]
                if classifier_score is not None:
                    fetches.append(classifier_score)
                values = session.run(fetches)
                address, label, vector = values[:3]
                score = values[3] if classifier_score is not None else None
            except tf.errors.OutOfRangeError:
                break
            for index, (account, target, row) in enumerate(
                    zip(address.reshape(-1), label.reshape(-1), vector)):
                account = int(account)
                if account not in sums:
                    sums[account] = row.astype(np.float64)
                    counts[account] = 1
                    labels[account] = float(target)
                    if score is not None:
                        score_sums[account] = float(score[index])
                else:
                    sums[account] += row
                    counts[account] += 1
                    labels[account] = max(labels[account], float(target))
                    if score is not None:
                        score_sums[account] += float(score[index])
                window_count += 1

    addresses = np.asarray(sorted(sums), dtype=np.int64)
    vectors = np.stack([sums[key] / counts[key] for key in addresses]).astype(np.float32)
    targets = np.asarray([labels[key] for key in addresses], dtype=np.int8)
    windows = np.asarray([counts[key] for key in addresses], dtype=np.int32)
    classifier_scores = np.asarray(
        [score_sums[key] / counts[key] for key in addresses], dtype=np.float32
    ) if classifier_score is not None else np.asarray([], dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    np.savez_compressed(
        args.output_file,
        addresses=addresses,
        labels=targets,
        vectors=vectors,
        windows=windows,
        classifier_scores=classifier_scores,
    )
    print("Exported %d accounts from %d windows; vectors=%s" % (
        len(addresses), window_count, vectors.shape
    ))


if __name__ == "__main__":
    main()
