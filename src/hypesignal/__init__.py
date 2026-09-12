"""HypeSignal: Offline-first AI-driven Social Media Analytics Framework."""

import uvicorn


def main() -> None:
    """CLI entry point for running the HypeSignal API server."""
    print("Launching HypeSignal Analytics API server at http://127.0.0.1:8000 ...")
    uvicorn.run("hypesignal.api:app", host="0.0.0.0", port=8000, reload=False)
