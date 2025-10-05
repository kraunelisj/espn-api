#!/usr/bin/env sh
set -euo pipefail

HOST="${MCPO_PROXY_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [ -n "${MCPO_PROXY_COMMAND:-}" ]; then
    exec sh -c "${MCPO_PROXY_COMMAND}"
fi

MCP_SERVER_COMMAND="${MCP_SERVER_COMMAND:-python -m espn_api.mcp_server}"

exec mcpo-proxy serve \
    --host "${HOST}" \
    --port "${PORT}" \
    --command "${MCP_SERVER_COMMAND}"

