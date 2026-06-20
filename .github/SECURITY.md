# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest (main) | Yes |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report security issues by emailing the maintainer directly. You will receive a response within 48 hours. If the issue is confirmed, a patch will be released as soon as possible.

## Security Design Notes

- All secrets (Langfuse keys, DB passwords, Grafana admin password) are loaded from environment variables — never hardcoded.
- Langfuse API key format is validated on startup (`sk-lf-` / `pk-lf-` prefix required).
- Prompt registry and baseline files are written with mode `0o600` (owner read/write only).
- The Docker Compose stack rejects startup if any required secret env var is missing (`:?` syntax).
- Guardrail extension provides keyword-based input/output filtering as a first line of defence against prompt injection or data leakage.
