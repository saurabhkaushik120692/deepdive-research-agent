#!/usr/bin/env bash
# One-time (idempotent) setup of an Amazon Linux 2023 EC2 instance for DeepDive.
# Run from a checkout of this repo on the instance:  bash deploy/ec2-bootstrap.sh
set -euo pipefail

COMPOSE_VERSION="v2.29.7"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> Docker"
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user || true

echo "==> Docker Compose plugin ${COMPOSE_VERSION}"
sudo mkdir -p /usr/local/lib/docker/cli-plugins
if ! docker compose version >/dev/null 2>&1; then
  sudo curl -fsSL \
    "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

echo "==> /opt/deepdive"
sudo mkdir -p /opt/deepdive/data
sudo chown -R 1000:1000 /opt/deepdive/data   # the container runs as uid 1000
sudo cp "$HERE/docker-compose.prod.yml" "$HERE/Caddyfile" /opt/deepdive/

if [ ! -f /opt/deepdive/.env ]; then
  echo
  echo "!! Create /opt/deepdive/.env (chmod 600) with at least:"
  echo "   MESH_API_KEY=...  TAVILY_API_KEY=...  DEMO_ACCESS_CODE=...  SITE_HOST=<eip-with-dashes>.nip.io"
fi
echo "==> done"
