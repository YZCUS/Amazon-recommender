"""Kafka producer placeholder.

This module defines a minimal interface for producing messages to Kafka. It is kept
lightweight to satisfy the repository structure and can be implemented later.
"""

from __future__ import annotations


def send_message(topic: str, payload: bytes) -> None:
    """Placeholder send function."""
    raise NotImplementedError("Kafka integration is not implemented in this skeleton.")


__all__ = ["send_message"]


