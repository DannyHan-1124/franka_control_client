#!/usr/bin/env bash
set -euo pipefail

# Open (or reuse) the SSH tunnel from the Franka client PC to the RTX 5090
# policy PC. Environment variables may override all connection parameters.

POLICY_SSH_TARGET="${POLICY_SSH_TARGET:-zhuoyue@141.3.54.19}"
LOCAL_POLICY_PORT="${LOCAL_POLICY_PORT:-17725}"
REMOTE_POLICY_HOST="${REMOTE_POLICY_HOST:-127.0.0.1}"
REMOTE_POLICY_PORT="${REMOTE_POLICY_PORT:-40023}"
CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/franka_policy_5090_${UID}.sock}"

if ssh -S "${CONTROL_SOCKET}" -O check "${POLICY_SSH_TARGET}" >/dev/null 2>&1; then
  echo "5090 policy tunnel is already running on 127.0.0.1:${LOCAL_POLICY_PORT}."
  exit 0
fi

# Remove only a stale SSH control socket; never disturb an unrelated listener.
if [[ -S "${CONTROL_SOCKET}" ]]; then
  unlink "${CONTROL_SOCKET}"
fi

echo "Opening 5090 policy tunnel:"
echo "  127.0.0.1:${LOCAL_POLICY_PORT} -> ${POLICY_SSH_TARGET}:${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}"

ssh -M -S "${CONTROL_SOCKET}" -fN \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LOCAL_POLICY_PORT}:${REMOTE_POLICY_HOST}:${REMOTE_POLICY_PORT}" \
  "${POLICY_SSH_TARGET}"

echo "5090 policy tunnel is ready on tcp://127.0.0.1:${LOCAL_POLICY_PORT}."
