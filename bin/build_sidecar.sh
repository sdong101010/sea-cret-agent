#!/bin/bash
# Build the Swift speech sidecar binary.
# The Python transcriber will run this automatically if the binary is missing.
set -euo pipefail
cd "$(dirname "$0")"
swiftc -O speech_sidecar.swift -o speech_sidecar
echo "Built: $(pwd)/speech_sidecar"
