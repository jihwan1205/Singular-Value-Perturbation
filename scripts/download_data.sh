#!/usr/bin/env bash
# Download the GSM8K-Aug training stream (385,620 problems, ~100 MB) used by
# SVP-V-GRPO. The six evaluation benchmarks are committed under data/.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="https://raw.githubusercontent.com/ModalityDance/LatentTTS/main/data/gsm_train.json"
if [ -s data/gsm_train.json ]; then
    echo "data/gsm_train.json already present"
    exit 0
fi
curl -L --fail -o data/gsm_train.json "$URL"
python -c "import json; d=json.load(open('data/gsm_train.json')); print(f'data/gsm_train.json: {len(d)} problems')"
