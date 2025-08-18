"""ngrok configuration placeholder.

Provide helpers for exposing local services (e.g., Kafka) through ngrok.
"""

from __future__ import annotations


def start_tunnel(port: int) -> None:
    """Placeholder for starting an ngrok tunnel to the given port."""
    raise NotImplementedError(
        "ngrok integration is not implemented in this skeleton.")


__all__ = ["start_tunnel"]
