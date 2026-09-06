#!/usr/bin/env bash
# Tải sẵn wheel trên HOST để build Docker offline.
# Lý do: mạng bên trong Docker build ở nhiều máy bị giới hạn băng thông rất nặng
# (đo được ~20-30kB/s so với >100MB/s trên host) -> `pip install` trong build gần
# như treo. Xem docs/TROUBLESHOOTING.md muc E02.
set -euo pipefail
cd "$(dirname "$0")/../services/api"
python3 -m pip download -d wheels -r requirements.txt \
    --python-version 3.11 --only-binary=:all:
echo "Đã tải $(ls wheels | wc -l) wheel vào services/api/wheels/"
