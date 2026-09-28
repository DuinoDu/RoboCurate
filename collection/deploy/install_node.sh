#!/usr/bin/env bash
# Install collection-node on a recording device as a user-level systemd service.
# No sudo needed when lingering is enabled for the user (loginctl show-user $USER -p Linger).
#
#   curl -fsSL http://<hub>:8422/install/install_node.sh | bash -s -- --hub http://<hub>:8422 --token <T> \
#       [--hub-key 'ssh-ed25519 AAAA… collection-hub'] [--data-root ~/holomotion-recordings/raw] \
#       [--allow-shutdown] [--allow-cleanup]
set -euo pipefail
HUB="" TOKEN="" KEY="" DATA_ROOT="$HOME/holomotion-recordings/raw" SHUT=false CLEAN=false
while [ $# -gt 0 ]; do
  case "$1" in
    --hub) HUB="$2"; shift ;;
    --token) TOKEN="$2"; shift ;;
    --hub-key) KEY="$2"; shift ;;
    --data-root) DATA_ROOT="$2"; shift ;;
    --allow-shutdown) SHUT=true ;;
    --allow-cleanup) CLEAN=true ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$HUB" ] && [ -n "$TOKEN" ] || { echo "--hub and --token are required" >&2; exit 2; }
APP="$HOME/.local/share/collection-node"; CONF="$HOME/.config/collection-node"; UNIT="$HOME/.config/systemd/user"
mkdir -p "$APP" "$CONF" "$UNIT" "$DATA_ROOT"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$SRC" ] && [ -f "$SRC/../node.py" ]; then cp "$SRC/../node.py" "$APP/node.py"
else curl -fsSL "$HUB/install/node.py" -o "$APP/node.py.tmp" && mv "$APP/node.py.tmp" "$APP/node.py"; fi
umask 077
python3 - "$CONF/config.json" "$HUB" "$TOKEN" "$DATA_ROOT" "$SHUT" "$CLEAN" <<'PY'
import json, sys
path, hub, token, root, shut, clean = sys.argv[1:]
json.dump({"hub": hub, "token": token, "data_root": root, "allow_shutdown": shut == "true",
           "allow_cleanup": clean == "true"}, open(path, "w"), indent=2)
PY
umask 022
cat > "$UNIT/collection-node.service" <<UNITEOF
[Unit]
Description=collection-node (reports recording state and data to collection-hub)
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 %h/.local/share/collection-node/node.py --config %h/.config/collection-node/config.json
Restart=always
RestartSec=5
Nice=10
IOSchedulingClass=idle
CPUQuota=5%
MemoryMax=64M

[Install]
WantedBy=default.target
UNITEOF
systemctl --user daemon-reload
systemctl --user enable --now collection-node.service
systemctl --user restart collection-node.service
if [ -n "$KEY" ]; then
  mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"; touch "$HOME/.ssh/authorized_keys"
  RR="$(command -v rrsync || echo /usr/bin/rrsync)"
  LINE="command=\"ionice -c3 nice -n19 $RR -ro $DATA_ROOT\",restrict $KEY"
  grep -v -F "$KEY" "$HOME/.ssh/authorized_keys" > "$HOME/.ssh/authorized_keys.new" || true
  echo "$LINE" >> "$HOME/.ssh/authorized_keys.new"
  mv "$HOME/.ssh/authorized_keys.new" "$HOME/.ssh/authorized_keys"; chmod 600 "$HOME/.ssh/authorized_keys"
  echo "hub key installed (read-only rrsync on $DATA_ROOT)"
fi
[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = yes ] || \
  echo "WARNING: lingering is off; ask an admin to run 'sudo loginctl enable-linger $USER' so the node starts at boot"
$SHUT && echo "NOTE: remote shutdown needs: echo '$USER ALL=(root) NOPASSWD: /sbin/shutdown' | sudo tee /etc/sudoers.d/collection-node"
sleep 2; systemctl --user --no-pager status collection-node.service | head -5
