#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

output=$1
mkdir -p "$output"

package() {
  local name=$1
  shift
  tar -czf - "$@" | split -b 1900M -d -a 2 - "$output/${name}.tar.gz.part-"
}

package zipzap-raw-data Data
package zipzap-label-free-data Model/data
package zipzap-benchmark-data Model/data_benchmark
package zipzap-model-artifacts \
  Model/project_results \
  Model/ckpt_pretrain \
  Model/ckpt_benchmark_pretrain \
  Model/ckpt_benchmark_finetuned \
  Model/ckpt_benchmark_frozen \
  Model/ckpt_benchmark_random_frozen

(
  cd "$output"
  sha256sum *.part-* > SHA256SUMS
  printf '%s\n' \
    'ZipZap extended workflow data and model artifacts' \
    '' \
    'Verify: sha256sum -c SHA256SUMS' \
    'Reassemble and extract one bundle:' \
    '  cat zipzap-raw-data.tar.gz.part-* > zipzap-raw-data.tar.gz' \
    '  tar -xzf zipzap-raw-data.tar.gz' \
    '' \
    'Repeat with zipzap-label-free-data, zipzap-benchmark-data, and' \
    'zipzap-model-artifacts. Extract from the repository root so the Data/' \
    'and Model/ paths are restored in their expected locations.' \
    > DOWNLOAD_AND_EXTRACT.txt
)
