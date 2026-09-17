#!/usr/bin/env bash
# An optional HTTP/mixed proxy applies to this command only.
set -euo pipefail
WARP_APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WARP_LOGIN_PROXY="${ROBOCURATE_HF_PROXY:-${HOLOCURATE_HF_PROXY:-}}"
WARP_AUTH_ACTION="${1:-login}"
case "$WARP_AUTH_ACTION" in
  login|whoami) ;;
  *) echo '用法：bash scripts/hf_login.sh [login|whoami]' >&2; exit 2 ;;
esac
if [ -n "$WARP_LOGIN_PROXY" ]; then
  case "$WARP_LOGIN_PROXY" in
    http://*|https://*) ;;
    *) echo 'ROBOCURATE_HF_PROXY must be an HTTP(S) proxy URL' >&2; exit 2 ;;
  esac
  exec env -u ALL_PROXY -u all_proxy \
    HTTP_PROXY="$WARP_LOGIN_PROXY" HTTPS_PROXY="$WARP_LOGIN_PROXY" \
    http_proxy="$WARP_LOGIN_PROXY" https_proxy="$WARP_LOGIN_PROXY" \
    "$WARP_APP_DIR/workspace/warp-env/bin/hf" auth "$WARP_AUTH_ACTION"
fi
exec "$WARP_APP_DIR/workspace/warp-env/bin/hf" auth "$WARP_AUTH_ACTION"
