#!/usr/bin/env bash
# Install a node on Raspberry Pi OS (64-bit, Bookworm). Run from the repo root as root:
#   sudo NODE_ID=node3 ./deploy/install-pi.sh
set -euo pipefail

NODE_ID="${NODE_ID:?set NODE_ID, e.g. NODE_ID=node3}"
PREFIX=/opt/edgecluster

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "expected a 64-bit OS (aarch64), got $(uname -m); onnxruntime has no 32-bit ARM wheels" >&2
  exit 1
fi

apt-get update
apt-get install -y --no-install-recommends python3-venv python3-dev sqlite3

id edge >/dev/null 2>&1 || useradd --system --home /var/lib/edgecluster --shell /usr/sbin/nologin edge
mkdir -p "$PREFIX" /etc/edgecluster /var/lib/edgecluster
cp -r edgecluster pyproject.toml README.md requirements "$PREFIX"/
python3 -m venv "$PREFIX/.venv"
"$PREFIX/.venv/bin/pip" install --upgrade pip
"$PREFIX/.venv/bin/pip" install -r "$PREFIX/requirements/runtime.txt"
"$PREFIX/.venv/bin/pip" install --no-deps "$PREFIX"

[[ -f /etc/edgecluster/cluster.yaml ]] || cp configs/cluster.example.yaml /etc/edgecluster/cluster.yaml
cat > /etc/edgecluster/node.env <<ENV
EDGE_NODE_ID=${NODE_ID}
EDGE_DATA_DIR=/var/lib/edgecluster
ENV
chown -R edge:edge /var/lib/edgecluster

cp deploy/systemd/edgecluster.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now edgecluster
systemctl --no-pager status edgecluster | head -5
echo "node ${NODE_ID} installed; edit /etc/edgecluster/cluster.yaml (peers, model_dir) and 'systemctl restart edgecluster'"
