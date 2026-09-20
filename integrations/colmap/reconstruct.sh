#!/usr/bin/env bash
# Build only COLMAP's sparse SfM model.  No Gaussian-training code belongs here.
set -euo pipefail

usage() {
  echo "Usage: $0 IMAGES_DIR OUTPUT_DIR [SINGLE_CAMERA=1]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

images_dir=$(realpath "$1")
output_dir=$2
single_camera=${3:-1}
colmap_bin=${COLMAP_BIN:-colmap}

[[ -d "$images_dir" ]] || { echo "Image directory does not exist: $images_dir" >&2; exit 1; }
command -v "$colmap_bin" >/dev/null || {
  echo "COLMAP is not on PATH. Activate gslab-rocm first." >&2
  exit 1
}

if [[ -e "$output_dir" ]] && [[ -n $(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  echo "Refusing to use non-empty output directory: $output_dir" >&2
  exit 1
fi

mkdir -p "$output_dir/sparse"
database="$output_dir/database.db"

"$colmap_bin" feature_extractor \
  --database_path "$database" \
  --image_path "$images_dir" \
  --ImageReader.single_camera "$single_camera" \
  --SiftExtraction.use_gpu 0

"$colmap_bin" exhaustive_matcher \
  --database_path "$database" \
  --SiftMatching.use_gpu 0

"$colmap_bin" mapper \
  --database_path "$database" \
  --image_path "$images_dir" \
  --output_path "$output_dir/sparse"

echo
echo "Sparse reconstruction complete. Inspect: $output_dir/sparse/0"
echo "This is a COLMAP artifact; hand it to a separate Gaussian trainer."
