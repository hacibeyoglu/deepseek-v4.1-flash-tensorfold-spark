#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for script in scripts/*.sh; do bash -n "$script"; done
python3 -m unittest discover -s tests -v
./stack run flash17-jay-exl3-2.8bpw --setup --dry-run -n 192.0.2.11,192.0.2.12 >/dev/null
