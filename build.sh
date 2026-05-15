#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
# Default build type and whether to build tests/docs (can be overridden via env)
BUILD_TYPE="${BUILD_TYPE:-Debug}"
ONNX_MLIR_BUILD_TESTS="${ONNX_MLIR_BUILD_TESTS:-OFF}"

# You can override these from environment variables when needed.
PROTOBUF_INSTALL_DIR="${PROTOBUF_INSTALL_DIR:-/data/dagongcheng/Common/protobuf/install}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/data/dagongcheng/Common/onnx-llvm-project/build}"
MLIR_DIR="${MLIR_DIR:-${LLVM_BUILD_DIR}/lib/cmake/mlir}"

PROTOC_BIN="${PROTOC_BIN:-${PROTOBUF_INSTALL_DIR}/bin/protoc}"
PROTOBUF_CMAKE_DIR="${PROTOBUF_CMAKE_DIR:-${PROTOBUF_INSTALL_DIR}/lib/cmake/protobuf}"

ONNX_MLIR_ENABLE_STABLEHLO="${ONNX_MLIR_ENABLE_STABLEHLO:-OFF}"
ONNX_MLIR_ENABLE_WERROR="${ONNX_MLIR_ENABLE_WERROR:-OFF}"
ONNX_USE_PROTOBUF_SHARED_LIBS="${ONNX_USE_PROTOBUF_SHARED_LIBS:-ON}"

if [[ ! -x "${PROTOC_BIN}" ]]; then
   echo "ERROR: protoc not found or not executable: ${PROTOC_BIN}" >&2
   exit 1
fi

if [[ ! -d "${MLIR_DIR}" ]]; then
   echo "ERROR: MLIR_DIR does not exist: ${MLIR_DIR}" >&2
   exit 1
fi

if [[ ! -d "${PROTOBUF_INSTALL_DIR}" ]]; then
   echo "ERROR: PROTOBUF_INSTALL_DIR does not exist: ${PROTOBUF_INSTALL_DIR}" >&2
   exit 1
fi

# Ensure local tools/libs are preferred over system defaults.
export PATH="${PROTOBUF_INSTALL_DIR}/bin:${LLVM_BUILD_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${PROTOBUF_INSTALL_DIR}/lib:${PROTOBUF_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}"

echo "[build.sh] Using protoc from: ${PROTOC_BIN}"
"${PROTOC_BIN}" --version
echo "[build.sh] Using MLIR_DIR: ${MLIR_DIR}"
echo "[build.sh] Using Protobuf CMake dir: ${PROTOBUF_CMAKE_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${ROOT_DIR}" -G "Ninja" \
   -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
   -DCMAKE_PREFIX_PATH="${PROTOBUF_INSTALL_DIR};${LLVM_BUILD_DIR};${CMAKE_PREFIX_PATH:-}" \
   -DMLIR_DIR="${MLIR_DIR}" \
   -DPROTOBUF_DIR="${PROTOBUF_CMAKE_DIR}" \
   -DProtobuf_PROTOC_EXECUTABLE="${PROTOC_BIN}" \
   -DONNX_USE_PROTOBUF_SHARED_LIBS="${ONNX_USE_PROTOBUF_SHARED_LIBS}" \
   -DProtobuf_USE_STATIC_LIBS=$([[ "${ONNX_USE_PROTOBUF_SHARED_LIBS}" == "ON" ]] && echo OFF || echo ON) \
   -DPROTOBUF_LIBRARIES="${PROTOBUF_INSTALL_DIR}/lib/libprotobuf.so" \
   -DPROTOBUF_INCLUDE_DIR="${PROTOBUF_INSTALL_DIR}/include" \
   -DONNX_MLIR_BUILD_TESTS="${ONNX_MLIR_BUILD_TESTS}" \
   -DLLVM_LIT_ARGS=-v \
   -DONNX_MLIR_ENABLE_STABLEHLO="${ONNX_MLIR_ENABLE_STABLEHLO}" \
   -DONNX_MLIR_ENABLE_WERROR="${ONNX_MLIR_ENABLE_WERROR}"

cmake --build . --config "${BUILD_TYPE}"
