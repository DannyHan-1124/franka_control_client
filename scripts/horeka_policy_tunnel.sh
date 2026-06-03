#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the direct-ZMQ policy tunnel.
# Prefer:
#   bash scripts/build_tunnel.sh <horeka_compute_node>

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/build_tunnel.sh" "$@"
