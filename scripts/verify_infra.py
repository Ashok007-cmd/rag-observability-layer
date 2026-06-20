#!/usr/bin/env python3
"""Infrastructure health checker for monitoring and observability stack.

Checks PostgreSQL, Langfuse, Prometheus, and Grafana status before running pipelines.
"""

import sys
import socket
import urllib.request
import urllib.error
import time

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError):
        return False


def check_http(url: str, expected_code: int = 200, timeout: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == expected_code
    except urllib.error.HTTPError as e:
        return e.code == expected_code
    except Exception:
        return False


def main():
    print("=" * 60)
    print("OBSERVABILITY INFRASTRUCTURE DIAGNOSTIC CHECK")
    print("=" * 60)

    checks = {
        "PostgreSQL (5432)": {"type": "port", "host": "localhost", "port": 5432},
        "Langfuse Server (3000)": {"type": "http", "url": "http://localhost:3000/api/health"},
        "Prometheus (9090)": {"type": "http", "url": "http://localhost:9090/-/healthy"},
        "Grafana UI (3001)": {"type": "http", "url": "http://localhost:3001/api/health"},
    }

    all_healthy = True
    for name, config in checks.items():
        print(f"Checking {name:25s} ... ", end="", flush=True)
        if config["type"] == "port":
            success = check_port(config["host"], config["port"])
        elif config["type"] == "http":
            success = check_http(config["url"])
        else:
            success = False

        if success:
            print(f"{GREEN}[ONLINE]{RESET}")
        else:
            print(f"{RED}[OFFLINE]{RESET}")
            all_healthy = False

    print("=" * 60)
    if all_healthy:
        print(f"{GREEN}STATUS: Observability stack is healthy and ready to receive traffic!{RESET}")
        sys.exit(0)
    else:
        print(f"{YELLOW}STATUS: One or more components are offline.{RESET}")
        print("Please run: scripts/start_monitoring_stack.sh to start the stack.")
        sys.exit(1)


if __name__ == "__main__":
    main()
