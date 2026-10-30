"""Disk-backed, bounded-memory replacement for the three preparation scripts."""

import argparse
import collections
import math
import os
import pickle
import random
import shutil

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import duckdb
import numpy as np
import tensorflow.compat.v1 as tf

from vocab import FreqVocab

tf.disable_v2_behavior()

SPECIAL_TOKENS = ["[MASK]", "[pad]", "[NO_USE]"]
CSV_COLUMNS = """{
    'hash':'VARCHAR','nonce':'VARCHAR','block_hash':'VARCHAR',
    'block_number':'BIGINT','transaction_index':'VARCHAR',
    'from_address':'VARCHAR','to_address':'VARCHAR','value':'HUGEINT',
    'gas':'VARCHAR','gas_price':'VARCHAR','input':'VARCHAR',
    'block_timestamp':'BIGINT','max_fee_per_gas':'VARCHAR',
    'max_priority_fee_per_gas':'VARCHAR','transaction_type':'VARCHAR'
}"""


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../Data")
    parser.add_argument("--phisher_accounts", default="")
    parser.add_argument("--output_dir", default="/home/jakku/llmfa-data")
    parser.add_argument("--work_dir", default="/home/jakku/llmfa-prep")
    parser.add_argument("--memory_limit", default="4GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dupe_factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rebuild_db", action="store_true")
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--skip_finetune", action="store_true")
    return parser.parse_args()


def quote(value):
    return "'" + value.replace("'", "''") + "'"


def csv_relation(path):
    return "read_csv(%s, header=false, columns=%s, null_padding=true, ignore_errors=true, parallel=true)" % (
        quote(os.path.abspath(path)), CSV_COLUMNS
    )


def source_select(path, source, direction):
    owner = "from_address" if direction == 1 else "to_address"
    counterparty = "to_address" if direction == 1 else "from_address"
    return """
        SELECT %s AS source, %d AS direction, %s AS owner,
               %s AS counterparty, block_number, block_timestamp,
               CAST(value AS DOUBLE) / 1000000000000.0 AS amount
        FROM %s
        WHERE %s IS NOT NULL AND %s <> ''
          AND %s IS NOT NULL AND %s <> ''
          AND block_timestamp IS NOT NULL
    """ % (
        quote(source), direction, owner, counterparty, csv_relation(path),
        owner, owner, counterparty, counterparty
    )


def build_database(connection, data_dir):
    files = [
        ("normal_eoa_transaction_out_slice_1000K.csv", "normal", 1),
        ("normal_eoa_transaction_in_slice_1000K.csv", "normal", 2),
        ("phisher_transaction_out.csv", "phisher", 1),
        ("phisher_transaction_in.csv", "phisher", 2),
    ]
    for filename, _, _ in files:
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    print("[1/4] Streaming CSV files into the disk-backed transaction table", flush=True)
    selects = [source_select(os.path.join(data_dir, *[name]), source, direction)
               for name, source, direction in files]
    connection.execute("DROP TABLE IF EXISTS raw_transactions")
    connection.execute("CREATE TABLE raw_transactions AS " + " UNION ALL ".join(selects))
    print("raw transactions:", connection.execute(
        "SELECT count(*) FROM raw_transactions").fetchone()[0], flush=True)

    print("[2/4] Aggregating repeated transactions", flush=True)
    connection.execute("DROP TABLE IF EXISTS aggregated_transactions")
    connection.execute("""
        CREATE TABLE aggregated_transactions AS
        WITH eligible AS (
            SELECT * FROM raw_transactions
            QUALIFY count(*) OVER (
                PARTITION BY source, direction, owner
            ) < 10000
        ), previous AS (
            SELECT *,
                lag(counterparty) OVER window_spec AS previous_counterparty,
                lag(block_timestamp) OVER window_spec AS previous_timestamp
            FROM eligible
            WINDOW window_spec AS (
                PARTITION BY source, direction, owner
                ORDER BY block_timestamp, block_number
            )
        ), boundaries AS (
            SELECT *, CASE
                WHEN previous_counterparty = counterparty
                 AND block_timestamp - previous_timestamp <= 259200
                THEN 0 ELSE 1 END AS new_group
            FROM previous
        ), grouped AS (
            SELECT *, sum(new_group) OVER (
                PARTITION BY source, direction, owner
                ORDER BY block_timestamp, block_number
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS transaction_group
            FROM boundaries
        )
        SELECT source, direction, owner, counterparty,
               min(block_number) AS block_number,
               min(block_timestamp) AS block_timestamp,
               sum(amount) AS amount,
               count(*)::INTEGER AS transaction_count
        FROM grouped
        GROUP BY source, direction, owner, counterparty, transaction_group
    """)
    connection.execute("DROP TABLE raw_transactions")

    print("[3/4] Selecting eligible accounts and bucketizing features", flush=True)
    connection.execute("DROP TABLE IF EXISTS sequences")
    connection.execute("""
        CREATE TABLE sequences AS
        WITH outgoing AS (
            SELECT DISTINCT source, owner
            FROM aggregated_transactions WHERE direction = 1
        ), eligible_rows AS (
            SELECT item.*
            FROM aggregated_transactions item
            INNER JOIN outgoing USING (source, owner)
        ), eligible_accounts AS (
            SELECT source, owner
            FROM eligible_rows
            GROUP BY source, owner
            HAVING sum(transaction_count) >= 5
        ), selected_source AS (
            SELECT owner,
                   CASE WHEN max(source = 'phisher') THEN 'phisher' ELSE 'normal' END AS source
            FROM eligible_accounts GROUP BY owner
        )
        SELECT row.owner, row.counterparty, row.block_number,
               row.block_timestamp, row.direction, row.transaction_count,
               CASE
                 WHEN row.amount = 0 THEN 1 WHEN row.amount <= 591 THEN 2
                 WHEN row.amount <= 6195 THEN 3 WHEN row.amount <= 21255 THEN 4
                 WHEN row.amount <= 50161 THEN 5 WHEN row.amount <= 100120 THEN 6
                 WHEN row.amount <= 208727 THEN 7 WHEN row.amount <= 508961 THEN 8
                 WHEN row.amount <= 1360574 THEN 9 WHEN row.amount <= 6500000 THEN 10
                 WHEN row.amount <= 143791433950 THEN 11 ELSE 12 END::INTEGER AS amount_bucket,
               CASE
                 WHEN row.transaction_count <= 7 THEN row.transaction_count
                 WHEN row.transaction_count <= 10 THEN 8
                 WHEN row.transaction_count <= 20 THEN 9 ELSE 10 END::INTEGER AS count_bucket
        FROM eligible_rows row
        INNER JOIN selected_source chosen USING (source, owner)
    """)
    connection.execute("DROP TABLE aggregated_transactions")
    connection.execute("CREATE INDEX sequence_owner ON sequences(owner)")
    stats = connection.execute("""
        SELECT count(*), sum(account_length), avg(account_length)
        FROM (
            SELECT owner, count(*) AS account_length FROM sequences GROUP BY owner
        ) accounts
    """).fetchone()
    print("accounts=%d aggregated_transactions=%d mean_length=%.2f" % stats, flush=True)
    print("[4/4] Database preparation complete", flush=True)


def build_vocab(connection, output_path):
    print("Building vocabulary", flush=True)
    query = """
        WITH counts AS (
            SELECT owner AS token, count(*)::BIGINT AS frequency
            FROM sequences GROUP BY owner
            UNION ALL
            SELECT counterparty AS token, count(*)::BIGINT AS frequency
            FROM sequences GROUP BY counterparty
        )
        SELECT token, sum(frequency)::BIGINT AS frequency
        FROM counts GROUP BY token ORDER BY frequency DESC, token
    """
    vocab = FreqVocab()
    vocab.special_tokens = list(SPECIAL_TOKENS)
    vocab.token_to_ids = {token: index + 1 for index, token in enumerate(SPECIAL_TOKENS)}
    vocab.frequency = [0] * len(SPECIAL_TOKENS)
    cursor = connection.execute(query)
    while True:
        rows = cursor.fetchmany(50000)
        if not rows:
            break
        for token, frequency in rows:
            vocab.token_to_ids[token] = len(vocab.token_to_ids) + 1
            vocab.frequency.append(int(frequency))
    vocab.token_count = len(vocab.token_to_ids) - len(SPECIAL_TOKENS)
    vocab.id_to_tokens = {identifier: token for token, identifier in vocab.token_to_ids.items()}
    vocab.vocab_words = list(vocab.token_to_ids)
    if len(vocab.vocab_words) > 2300000:
        raise ValueError("Vocabulary exceeds configured size: %d" % len(vocab.vocab_words))
    with open(output_path, "wb") as handle:
        pickle.dump(vocab, handle, protocol=2)
    print("vocabulary tokens:", len(vocab.vocab_words), flush=True)
    return vocab


def account_groups(connection):
    cursor = connection.execute("""
        SELECT owner, counterparty, block_number, block_timestamp,
               amount_bucket, direction, count_bucket
        FROM sequences
        ORDER BY owner, block_timestamp DESC, block_number DESC, counterparty
    """)
    current_owner, current_rows = None, []
    while True:
        batch = cursor.fetchmany(50000)
        if not batch:
            break
        for row in batch:
            owner = row[0]
            transaction = [row[1], int(row[2]), int(row[3]), int(row[4]),
                           int(row[5]), int(row[6])]
            if current_owner is not None and owner != current_owner:
                yield current_owner, current_rows
                current_rows = []
            current_owner = owner
            current_rows.append(transaction)
    if current_owner is not None:
        yield current_owner, current_rows


def windows(owner, rows, max_transactions, step, rng, deduplicate):
    if deduplicate:
        seen, reduced = set(), []
        for row in rows:
            if row[0] not in seen:
                seen.add(row[0])
                reduced.append(row)
        rows = reduced
    if len(rows) <= max_transactions:
        yield owner, rows
        return
    starts = list(range(len(rows) - max_transactions, 0, -step))
    starts.append(0)
    if len(starts) > 500:
        starts = list(rng.sample(starts, 500))
    else:
        starts = starts[::-1]
    for start in starts:
        yield owner, rows[start:start + max_transactions]


def int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def positions(timestamps):
    result, previous, position = [0], None, 0
    for timestamp in timestamps:
        if timestamp != previous:
            position += 1
            previous = timestamp
        result.append(position)
    return result


def base_features(owner, rows, vocab, length):
    tokens = [owner] + [row[0] for row in rows]
    ids = [vocab.token_to_ids[token] for token in tokens]
    counts = [0] + [row[5] for row in rows]
    values = [0] + [row[3] for row in rows]
    io_flags = [0] + [row[4] for row in rows]
    token_positions = positions([row[2] for row in rows])
    mask = [1] * len(ids)
    padding = length - len(ids)
    return (
        ids + [0] * padding, token_positions + [0] * padding,
        counts + [0] * padding, io_flags + [0] * padding,
        values + [0] * padding, mask + [0] * padding,
    )


def write_pretrain(connection, vocab, path, dupe_factor, seed):
    print("Streaming pretraining TFRecord", flush=True)
    rng = random.Random(seed)
    writer = tf.python_io.TFRecordWriter(path)
    total, length, maximum_predictions = 0, 29, math.ceil(29 * 0.8)
    try:
        for account_index, (owner, rows) in enumerate(account_groups(connection), 1):
            for _, selected in windows(owner, rows, 28, 17, rng, True):
                original = [owner] + [row[0] for row in selected]
                base = base_features(owner, selected, vocab, length)
                for _ in range(dupe_factor):
                    candidates = list(range(len(original)))
                    rng.shuffle(candidates)
                    count = min(maximum_predictions, max(1, int(len(original) * 0.8)))
                    chosen = sorted(candidates[:count])
                    input_ids = list(base[0])
                    labels = []
                    for index in chosen:
                        labels.append(vocab.token_to_ids[original[index]])
                        input_ids[index] = vocab.token_to_ids["[MASK]"]
                    feature = collections.OrderedDict()
                    feature["address"] = int_feature([vocab.token_to_ids[owner]])
                    feature["input_ids"] = int_feature(input_ids)
                    feature["input_positions"] = int_feature(base[1])
                    feature["input_counts"] = int_feature(base[2])
                    feature["input_io_flags"] = int_feature(base[3])
                    feature["input_values"] = int_feature(base[4])
                    feature["input_mask"] = int_feature(base[5])
                    feature["masked_lm_positions"] = int_feature(chosen + [0] * (maximum_predictions - len(chosen)))
                    feature["masked_lm_ids"] = int_feature(labels + [0] * (maximum_predictions - len(labels)))
                    feature["masked_lm_weights"] = float_feature([1.0] * len(labels) + [0.0] * (maximum_predictions - len(labels)))
                    writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
                    total += 1
            if account_index % 10000 == 0:
                print("pretrain accounts=%d records=%d" % (account_index, total), flush=True)
    finally:
        writer.close()
    print("pretraining records:", total, flush=True)


def load_phishers(data_dir, explicit_path=""):
    path = explicit_path or os.path.join(data_dir, "phisher_account.txt")
    with open(path, encoding="utf-8") as handle:
        return {line.strip().split(",")[0] for line in handle if line.strip()}


def write_finetune(connection, vocab, output_dir, data_dir, phisher_accounts, seed):
    print("Streaming finetuning TFRecords", flush=True)
    accounts = [row[0] for row in connection.execute(
        "SELECT DISTINCT owner FROM sequences ORDER BY owner").fetchall()]
    rng = random.Random(seed)
    rng.shuffle(accounts)
    split = round(len(accounts) * 0.7)
    training = set(accounts[:split])
    phishers = load_phishers(data_dir, phisher_accounts)
    paths = {
        True: os.path.join(output_dir, "finetune_train.tfrecord"),
        False: os.path.join(output_dir, "finetune_test.tfrecord"),
    }
    writers = {key: tf.python_io.TFRecordWriter(value) for key, value in paths.items()}
    totals = {True: 0, False: 0}
    try:
        for account_index, (owner, rows) in enumerate(account_groups(connection), 1):
            is_training = owner in training
            label = 1.0 if owner in phishers else 0.0
            for _, selected in windows(owner, rows, 99, 60, rng, False):
                base = base_features(owner, selected, vocab, 100)
                feature = collections.OrderedDict()
                feature["address"] = int_feature([vocab.token_to_ids[owner]])
                feature["label"] = float_feature([label])
                feature["input_ids"] = int_feature(base[0])
                feature["input_positions"] = int_feature(base[1])
                feature["input_counts"] = int_feature(base[2])
                feature["input_io_flags"] = int_feature(base[3])
                feature["input_values"] = int_feature(base[4])
                feature["input_mask"] = int_feature(base[5])
                writers[is_training].write(tf.train.Example(
                    features=tf.train.Features(feature=feature)).SerializeToString())
                totals[is_training] += 1
            if account_index % 10000 == 0:
                print("finetune accounts=%d train=%d test=%d" % (
                    account_index, totals[True], totals[False]), flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    print("finetuning records: train=%d test=%d" % (
        totals[True], totals[False]), flush=True)


def main():
    args = arguments()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    database = os.path.join(args.work_dir, "zipzap.duckdb")
    if args.rebuild_db and os.path.exists(database):
        os.remove(database)
    connection = duckdb.connect(database)
    connection.execute("SET memory_limit = %s" % quote(args.memory_limit))
    connection.execute("SET threads = %d" % args.threads)
    connection.execute("SET temp_directory = %s" % quote(os.path.join(args.work_dir, "duckdb-tmp")))
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if "sequences" not in tables:
        build_database(connection, args.data_dir)
    vocab_path = os.path.join(args.output_dir, "vocab")
    vocab = build_vocab(connection, vocab_path)
    if not args.skip_pretrain:
        write_pretrain(connection, vocab, os.path.join(args.output_dir, "train.tfrecord"),
                       args.dupe_factor, args.seed)
    if not args.skip_finetune:
        write_finetune(connection, vocab, args.output_dir, args.data_dir,
                       args.phisher_accounts, args.seed)
    connection.close()
    print("Preparation complete:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
