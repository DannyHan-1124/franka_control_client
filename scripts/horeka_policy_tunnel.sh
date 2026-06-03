#!/usr/bin/env bash
set -euo pipefail

# Build the SSH forwarding needed when the PI0.5 policy node runs on a HoreKa
# compute node and the robot/camera inference script runs on the lab machine.
#
# Usage, on the lab computer:
#   bash scripts/horeka_policy_tunnel.sh hkn0402 36611
#
# Then point the lab-side inference script/client at:
#   127.0.0.1:7725
#
# Important:
#   - HOREKA_NODE is the compute node printed by your interactive session.
#   - HOREKA_POLICY_PORT is the ServiceManager REP port printed by the policy
#     node running on HoreKa.
#   - LAB_POLICY_PORT is a local-only port on the lab computer. Keep it different
#     from robot/gripper/camera ServiceManager ports if those are also local.

HOREKA_LOGIN="${HOREKA_LOGIN:-utphd@horeka.scc.kit.edu}"
LAB_POLICY_HOST="${LAB_POLICY_HOST:-127.0.0.1}"
LAB_POLICY_PORT="${LAB_POLICY_PORT:-7725}"

HOREKA_NODE="${1:-${HOREKA_NODE:-}}"
HOREKA_POLICY_PORT="${2:-${HOREKA_POLICY_PORT:-}}"

if [[ -z "${HOREKA_NODE}" || -z "${HOREKA_POLICY_PORT}" ]]; then
  cat >&2 <<EOF
Usage:
  bash scripts/horeka_policy_tunnel.sh <horeka_compute_node> <horeka_policy_port>

Example:
  bash scripts/horeka_policy_tunnel.sh hkn0402 36611

Environment overrides:
  HOREKA_LOGIN=${HOREKA_LOGIN}
  LAB_POLICY_HOST=${LAB_POLICY_HOST}
  LAB_POLICY_PORT=${LAB_POLICY_PORT}
EOF
  exit 2
fi

echo "Opening tunnel:"
echo "  ${LAB_POLICY_HOST}:${LAB_POLICY_PORT} -> ${HOREKA_NODE}:${HOREKA_POLICY_PORT} via ${HOREKA_LOGIN}"

ssh -fN \
  -o ExitOnForwardFailure=yes \
  -L "${LAB_POLICY_HOST}:${LAB_POLICY_PORT}:${HOREKA_NODE}:${HOREKA_POLICY_PORT}" \
  "${HOREKA_LOGIN}"

cat <<EOF

Tunnel is up.

Run the lab-side inference/client with the HoreKa policy endpoint set to:
  ${LAB_POLICY_HOST}:${LAB_POLICY_PORT}

For example, if your script reads PI05_POLICY_HOST / PI05_POLICY_PORT:
  PI05_POLICY_HOST=${LAB_POLICY_HOST} PI05_POLICY_PORT=${LAB_POLICY_PORT} bash scripts/inference.sh

If your script takes command-line flags, use the equivalent:
  --policy-host ${LAB_POLICY_HOST} --policy-port ${LAB_POLICY_PORT}

EOF
