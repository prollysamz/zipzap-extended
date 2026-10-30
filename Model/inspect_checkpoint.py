"""Inspect a checkpoint without inferring label independence from tensor names."""
import argparse
import json
from research_protocol import read_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="TensorFlow checkpoint prefix (without .index)")
    args = parser.parse_args()
    import tensorflow.compat.v1 as tf
    variables = tf.train.list_variables(args.checkpoint)
    provenance = read_provenance(args.checkpoint)
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "variable_count": len(variables),
        "contains_mlp": any(name.startswith("MLP/") for name, _ in variables),
        "contains_shared_encoder": any("/shared_layer/" in name for name, _ in variables),
        "eligible_as_label_independent_encoder": provenance.get("verified", False)
        and provenance.get("encoder_label_use") is False,
        "provenance": provenance,
        "note": "Absent MLP weights do not prove that the encoder never used labels. "
                "Manifests record training lineage; independent datasets and logs must corroborate it.",
    }, indent=2))


if __name__ == "__main__":
    main()
