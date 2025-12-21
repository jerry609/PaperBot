#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

echo "OK: dev deps installed"

