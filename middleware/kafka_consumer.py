"""Kafka consumer placeholder.

This module defines a minimal interface for consuming messages from Kafka.
"""

from __future__ import annotations


def consume(topic: str):
    """Placeholder consume generator."""
    raise NotImplementedError("Kafka integration is not implemented in this skeleton.")


__all__ = ["consume"]


