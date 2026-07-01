#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting monitoring stack (Langfuse + Prometheus + Grafana)..."
docker compose -f "$PROJECT_DIR/infra/docker-compose.yml" up -d

echo "Waiting for services to be healthy..."
sleep 5

echo ""
echo "Monitoring stack is running:"
echo "  Langfuse:   http://localhost:3000"
echo "  Prometheus:  http://localhost:9090"
echo "  Grafana:    http://localhost:3001 (admin / \$GF_ADMIN_PASSWORD from your .env)"
echo ""
echo "To stop: docker compose -f $PROJECT_DIR/infra/docker-compose.yml down"
