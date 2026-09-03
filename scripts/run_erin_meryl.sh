#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKSPACE="${CHANNEL_COMPRESSOR_WORKSPACE:-${CHANNEL80_WORKSPACE:-$ROOT/erin-meryl-corpus}}"
PROVIDERS="${CHANNEL_COMPRESSOR_PROVIDERS:-${CHANNEL80_PROVIDERS:-youtube,ytdlp,whisper}}"
MODE="${CHANNEL_COMPRESSOR_ANALYSIS_MODE:-${CHANNEL80_ANALYSIS_MODE:-auto}}"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'

channel-compressor run \
  'https://www.youtube.com/@erinmerylstudy/videos' \
  --workspace "$WORKSPACE" \
  --profile profiles/sam.yaml \
  --providers "$PROVIDERS" \
  --analysis-mode "$MODE" \
  --target-coverage 0.80 \
  --max-minutes 120

echo
echo "Report: $WORKSPACE/outputs/report.html"
