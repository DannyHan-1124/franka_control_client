#!/usr/bin/env bash
set -euo pipefail

# Build the SSH tunnel used by direct-ZMQ PI0.5 policy inference on HoreKa 2.
#
# Run this on the lab computer after scripts/start_policy_node.sh is running in
# an interactive session on the allocated HoreKa 2 compute node.
# Usage:
#   bash scripts/build_tunnel.sh hk2n001

# HoreKa 2 gives KIT accounts the "ka_" site prefix. HOREKA_LOGIN and
# HOREKA_NODE remain supported so existing local configuration keeps working.
HOREKA2_LOGIN="${HOREKA2_LOGIN:-${HOREKA_LOGIN:-ka_utphd@hk2-x86.scc.kit.edu}}"
HOREKA2_NODE="${1:-${HOREKA2_NODE:-${HOREKA_NODE:-}}}"
LAB_POLICY_PORT="${LAB_POLICY_PORT:-17725}"
HOREKA_POLICY_PORT="${HOREKA_POLICY_PORT:-40023}"

if [[ -z "${HOREKA2_NODE}" ]]; then
  cat >&2 <<EOF
Usage:
  bash scripts/build_tunnel.sh <horeka2_compute_node>

Example:
  bash scripts/build_tunnel.sh hk2n001

On the HoreKa 2 interactive session, get the node name with:
  hostname

Environment overrides:
  HOREKA2_LOGIN=${HOREKA2_LOGIN}
  LAB_POLICY_PORT=${LAB_POLICY_PORT}
  HOREKA_POLICY_PORT=${HOREKA_POLICY_PORT}
EOF
  exit 2
fi

echo "Opening tunnel:"
echo "  lab 127.0.0.1:${LAB_POLICY_PORT} -> ${HOREKA2_NODE}:${HOREKA_POLICY_PORT}"
echo "  via ${HOREKA2_LOGIN}"

ssh -fN \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LAB_POLICY_PORT}:${HOREKA2_NODE}:${HOREKA_POLICY_PORT}" \
  "${HOREKA2_LOGIN}"

cat <<EOF

Tunnel is up.

Check it on the lab computer:
  nc -vz 127.0.0.1 ${LAB_POLICY_PORT}

Then run:
  bash scripts/inference.sh

EOF
