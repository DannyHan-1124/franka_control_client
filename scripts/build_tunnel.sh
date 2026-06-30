#!/usr/bin/env bash
set -euo pipefail

# Build the one SSH tunnel used by streaming-ZMQ PI0.5 policy inference.
#
# Run on the lab computer after scripts/test.sh is running on the HoreKa GPU node.
# Usage:
#   bash scripts/build_tunnel.sh hkn0401

HOREKA_LOGIN="${HOREKA_LOGIN:-utphd@horeka.scc.kit.edu}"
HOREKA_NODE="${1:-${HOREKA_NODE:-}}"
LAB_POLICY_PORT="${LAB_POLICY_PORT:-17826}"
HOREKA_POLICY_PORT="${HOREKA_POLICY_PORT:-40124}"

if [[ -z "${HOREKA_NODE}" ]]; then
  cat >&2 <<EOF
Usage:
  bash scripts/build_tunnel.sh <horeka_compute_node>

Example:
  bash scripts/build_tunnel.sh hkn0401

Environment overrides:
  HOREKA_LOGIN=${HOREKA_LOGIN}
  LAB_POLICY_PORT=${LAB_POLICY_PORT}
  HOREKA_POLICY_PORT=${HOREKA_POLICY_PORT}
EOF
  exit 2
fi

echo "Opening tunnel:"
echo "  lab 127.0.0.1:${LAB_POLICY_PORT} -> ${HOREKA_NODE}:127.0.0.1:${HOREKA_POLICY_PORT}"

ssh -fN \
  -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:${LAB_POLICY_PORT}:${HOREKA_NODE}:${HOREKA_POLICY_PORT}" \
  "${HOREKA_LOGIN}"

cat <<EOF

Tunnel is up.

Check it on the lab computer:
  nc -vz 127.0.0.1 ${LAB_POLICY_PORT}

Then run:
  bash scripts/inference.sh

EOF
