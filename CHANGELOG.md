# Changelog

All notable repository-level changes should be documented here. This project follows a lightweight changelog discipline suitable for a research benchmark; versions refer to the package version in `pyproject.toml` unless otherwise noted.

## Unreleased

### Documentation and research metadata

- Professionalized the repository landing page and project structure documentation.
- Added the AI2M4RI 2026 paper as the preferred scholarly citation.
- Added a reproducibility protocol for publication-quality benchmark runs.
- Added contribution guidance for changes that affect physics, constraints, metrics, seeds, safety logic, or benchmark comparability.
- Added a security policy and explicit research/industrial-use boundaries.
- Added a pull request template focused on scientific validity and reproducibility.

### Packaging and automation

- Improved Python package metadata and project links.
- Added the `eaf-twin` command-line entry point.
- Strengthened CI checks for dependency consistency, syntax/import integrity, tests, and simulation smoke validation.

## 0.1.0 - 2026

Initial research benchmark release with:

- empirical, first-principles, and enhanced-hybrid EAF digital-twin models;
- classical and learning-based controller families;
- agent training and evaluation runners;
- safety-aware simulation logic;
- Docker execution support;
- automated tests and benchmark reporting.
