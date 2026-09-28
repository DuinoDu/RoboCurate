#!/usr/bin/env bash
# Install collection-hub on the storage server as a user-level systemd service.
#   collection/deploy/install_hub.sh [--mirror-root DIR] [--reserve-gb N]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MIRROR="$HOME/holomotion_collection_data/raw" RESERVE=100
while [ $# -gt 0 ]; do
  case "$1" in
    --mirror-root) MIRROR="$2"; shift ;;
    --reserve-gb) RESERVE="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
CONF="$HOME/.config/collection-hub"; UNIT="$HOME/.config/systemd/user"
mkdir -p "$CONF" "$UNIT" "$MIRROR"
[ -f "$CONF/id_ed25519" ] || ssh-keygen -q -t ed25519 -N "" -C "collection-hub@$(hostname)" -f "$CONF/id_ed25519"
if [ ! -f "$CONF/config.json" ]; then
  cat > "$CONF/config.json" <<JSON
{
  "mirror_root": "$MIRROR",
  "db": "$HOME/.local/share/collection-hub/hub.sqlite3",
  "ssh_key": "$CONF/id_ed25519",
  "reserve_gb": $RESERVE,
  "node_listen": "0.0.0.0:8422",
  "admin_listen": "127.0.0.1:8423"
}
JSON
fi
cat > "$UNIT/collection-hub.service" <<UNITEOF
[Unit]
Description=collection-hub (device status, sync, disk monitor)
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 $REPO/collection/hub.py --config %h/.config/collection-hub/config.json serve
Restart=always
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=default.target
UNITEOF
systemctl --user daemon-reload
systemctl --user enable --now collection-hub.service
systemctl --user restart collection-hub.service
echo "hub public key (paste into install_node.sh --hub-key):"
cat "$CONF/id_ed25519.pub"
[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = yes ] || \
  echo "WARNING: lingering is off; 'sudo loginctl enable-linger $USER' is needed for the hub to start at boot"
