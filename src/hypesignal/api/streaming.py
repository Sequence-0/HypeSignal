"""Server-Sent Events (SSE) Broadcaster with explicit backpressure management."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Set, Union

from fastapi import Request

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Manages event fan-out to connected SSE clients with bounded-queue backpressure."""

    def __init__(self, max_queue_size: int = 100) -> None:
        """Initialize EventBroadcaster with bounded queue size.
        
        Args:
            max_queue_size: Maximum unconsumed events buffered per subscriber.
        """
        self.max_queue_size = max_queue_size
        self._subscribers: Set[asyncio.Queue[Dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    @property
    def subscriber_count(self) -> int:
        """Return the number of currently active subscribers."""
        return len(self._subscribers)

    async def subscribe(self) -> asyncio.Queue[Dict[str, Any]]:
        """Create and register a bounded queue for a newly connected client."""
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=self.max_queue_size)
        async with self._lock:
            self._subscribers.add(queue)
        logger.debug("New client subscribed to SSE stream (total: %d)", len(self._subscribers))
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        """Unregister a client queue upon disconnect."""
        async with self._lock:
            self._subscribers.discard(queue)
        logger.debug("Client unsubscribed from SSE stream (total: %d)", len(self._subscribers))

    def broadcast(self, event: Union[Dict[str, Any], str], event_type: str = "message") -> int:
        """Broadcast an event payload to all active client queues with drop-oldest backpressure.
        
        Args:
            event: Event data (dict or string).
            event_type: SSE event name.
            
        Returns:
            Number of subscribers the event was pushed to.
        """
        if not self._subscribers:
            return 0

        payload: Dict[str, Any] = {
            "event": event_type,
            "data": event if isinstance(event, dict) else {"message": str(event)},
        }

        delivered = 0
        # Iterate over a snapshot of subscribers
        for q in list(self._subscribers):
            if q.full():
                try:
                    # Drop oldest unconsumed event to avoid memory leak and blocking
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            try:
                q.put_nowait(payload)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning("Subscriber queue was full after drop-oldest attempt.")

        return delivered

    async def sse_generator(
        self,
        request: Request,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronously yield formatted SSE event dictionaries while monitoring client disconnect.
        
        Args:
            request: FastAPI request to detect disconnection.
            heartbeat_interval: Seconds between keep-alive pings.
            
        Yields:
            Dict containing 'event' and 'data' suitable for EventSourceResponse.
        """
        queue = await self.subscribe()
        try:
            while True:
                is_disconnected = request.is_disconnected()
                if asyncio.iscoroutine(is_disconnected) or hasattr(is_disconnected, "__await__"):
                    is_disconnected = await is_disconnected

                if is_disconnected:
                    logger.debug("SSE client disconnected.")
                    break

                try:
                    item = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
                    data_str = json.dumps(item.get("data", {}))
                    yield {
                        "event": item.get("event", "message"),
                        "data": data_str,
                    }
                except asyncio.TimeoutError:
                    # Send periodic keep-alive comment / ping
                    yield {
                        "event": "ping",
                        "data": json.dumps({"type": "heartbeat"}),
                    }
        except asyncio.CancelledError:
            logger.debug("SSE stream task cancelled.")
            raise
        finally:
            await self.unsubscribe(queue)
