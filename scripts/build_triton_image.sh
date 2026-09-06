#!/usr/bin/env bash
# Build image Triton NHẸ: chỉ giữ backend ONNX Runtime thay vì mọi backend.
# Ảnh chính thức đầy đủ ~27GB (gói cả TensorFlow, PyTorch, TensorRT...);
# bản compose lại chỉ còn ~19GB mà vẫn là Triton thật, chạy được GPU.
set -euo pipefail
VERSION="${TRITON_VERSION:-24.12}"
WORKDIR="${WORKDIR:-/tmp/triton-src}"

if docker image inspect tritonserver-ner-slim >/dev/null 2>&1; then
    echo "Image tritonserver-ner-slim đã có sẵn, bỏ qua. Xoá nó nếu muốn build lại."
    exit 0
fi

rm -rf "$WORKDIR"
git clone --depth 1 --branch "r${VERSION}" \
    https://github.com/triton-inference-server/server.git "$WORKDIR"
cd "$WORKDIR"
python3 compose.py --backend onnxruntime --container-version "$VERSION" \
    --output-name tritonserver-ner-slim
echo "Xong. Kiểm tra: docker images | grep tritonserver-ner-slim"
