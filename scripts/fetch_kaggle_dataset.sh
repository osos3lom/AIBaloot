#!/usr/bin/env bash
# Download and unpack a Kaggle playing-card dataset for Baloot training.
#
# Requires a Kaggle account: put your kaggle.json at ~/.kaggle/kaggle.json, or
# export KAGGLE_USERNAME and KAGGLE_KEY. The script never stores credentials.
#
# Usage:
#   scripts/fetch_kaggle_dataset.sh [destination-dir] [dataset-slug]
#
# Then, in the repo:
#   uv run hakim-vision dataset inspect --source <destination-dir>/<name>
#   uv run hakim-vision dataset remap  --source <...> --output data/baloot-dataset
#   uv run hakim-vision train --data data/baloot-dataset/data.yaml
set -euo pipefail

DEST="${1:-data/downloads}"
SLUG="${2:-andy8744/playing-cards-object-detection-dataset}"
NAME="$(basename "$SLUG")"
ARCHIVE="$DEST/$NAME.zip"

mkdir -p "$DEST"

if command -v kaggle >/dev/null 2>&1; then
  echo "Downloading $SLUG with the kaggle CLI…"
  kaggle datasets download -d "$SLUG" -p "$DEST"
else
  echo "kaggle CLI not found; using the public API endpoint (needs a browser login cookie or KAGGLE_KEY)."
  AUTH=()
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    AUTH=(-u "${KAGGLE_USERNAME}:${KAGGLE_KEY}")
  fi
  curl -L "${AUTH[@]}" -o "$ARCHIVE" \
    "https://www.kaggle.com/api/v1/datasets/download/$SLUG"
fi

ARCHIVE="$(ls -t "$DEST"/*.zip | head -1)"
echo "Unpacking $ARCHIVE…"
python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  "$ARCHIVE" "$DEST/$NAME"

echo
echo "Done: $DEST/$NAME"
echo "Next:"
echo "  uv run hakim-vision dataset inspect --source $DEST/$NAME"
echo "  uv run hakim-vision studio      # same steps with a UI"
