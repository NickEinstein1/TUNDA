"""Tracing and structured logging utilities."""

import json
import uuid
from typing import Any, Dict


def new_trace_id() -> str:
    return uuid.uuid4().hex


def log_event(logger, event: str, trace_id: str, **fields: Dict[str, Any]):
    payload = {"event": event, "trace_id": trace_id, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))
