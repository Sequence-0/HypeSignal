"""HypeSignal: Offline-first AI-driven Social Media Analytics Framework."""

import uvicorn


def main() -> None:
    """CLI entry point for running the HypeSignal API server."""
    from hypesignal.config import load_env

    import os

    load_env()
    port = int(os.getenv("PORT", os.getenv("HYPESIGNAL_PORT", "8080")))
    print(f"Launching HypeSignal Analytics API server at http://127.0.0.1:{port} ...")
    uvicorn.run("hypesignal.api:app", host="0.0.0.0", port=port, reload=False)
