import asyncio
import json
from datetime import datetime, timezone


class EventCollector:
    def __init__(self):
        self.clients = set()

    # incoming events are added to the queue
    async def publish(self, event: dict):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        for queue in list(self.clients):
            await queue.put(event)

    def publish_nowait(self, event: dict):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        for queue in list(self.clients):
            queue.put_nowait(event)

    # enables the retrieval of events from the queue
    async def subscribe(self):
        queue = asyncio.Queue()
        self.clients.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self.clients.discard(queue)


monitor = EventCollector()
