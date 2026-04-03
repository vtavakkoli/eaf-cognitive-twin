from __future__ import annotations


def model_status(issues: list[str]) -> str:
    return "ok" if not issues else "warning"
