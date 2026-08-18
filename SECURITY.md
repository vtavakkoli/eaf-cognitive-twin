# Security Policy

EAF Cognitive Twin is a research benchmark and simulation platform. It is **not** a production-certified EAF control system.

## Supported versions

Security fixes are applied to the current `main` branch. Research branches and historical commits may not receive security updates.

## Reporting a vulnerability

Please do not publish exploit details, credentials, confidential plant information, or other sensitive material in a public issue.

For a suspected security vulnerability, contact the repository maintainers privately using the author contact information in `CITATION.cff`. Include:

- a concise description of the issue;
- affected component and version/commit;
- reproduction steps or proof of concept;
- potential impact;
- suggested mitigation, if known.

For ordinary bugs, numerical issues, benchmark discrepancies, or reproducibility questions that do not expose sensitive information, use the public GitHub issue tracker.

## Security boundaries

The repository assumes a research environment. Users are responsible for securing:

- host and container runtime configuration;
- model/checkpoint files and external data;
- any integrations added around the benchmark;
- credentials, tokens, registries, and artifact storage;
- network exposure of locally added services.

Do not connect this software directly to industrial equipment or safety-critical control interfaces without independent engineering, cybersecurity, process-safety, and operational validation.

## Sensitive and industrial data

Do not commit confidential plant data, personal data, credentials, proprietary process parameters, or information you are not authorized to disclose. When reporting a problem, minimize and sanitize attached logs and configurations.
