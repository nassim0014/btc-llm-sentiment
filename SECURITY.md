# Security Policy

## Supported Versions

Only the latest `main` branch is supported with security updates.

## Reporting a Vulnerability

Email: nassim@kinzoils.com

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 48 hours. If the vulnerability is
confirmed, a fix will be released within 7 days for critical issues.

## Security Measures

This repo uses:
- **Bandit** SAST scanning in CI (on every push/PR)
- **pip-audit** dependency vulnerability scanning
- **gitleaks** secret detection
- **Trivy** container image scanning
- JWT-based authentication on API endpoints (where applicable)
- Rate limiting on API endpoints (where applicable)

## Dependencies

Dependencies are pinned to major versions (`>=X.Y,<X+1.0`) to balance
security fixes with stability. See `requirements.txt` for the full list.
