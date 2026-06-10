"""WebSocket handler for real-time engine progress streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# Global connection manager
class ConnectionManager:
    """Manages WebSocket connections per analysis."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, analysis_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(analysis_id, []).append(websocket)
        logger.info(f"WebSocket connected for analysis {analysis_id}")

    def disconnect(self, analysis_id: str, websocket: WebSocket):
        if analysis_id in self.active_connections:
            self.active_connections[analysis_id] = [
                ws for ws in self.active_connections[analysis_id] if ws != websocket
            ]
            if not self.active_connections[analysis_id]:
                del self.active_connections[analysis_id]

    async def broadcast(self, analysis_id: str, engine: str, progress: int, message: str, status: str = "running"):
        """Broadcast progress to all connected clients for an analysis."""
        data = json.dumps({
            "engine": engine,
            "progress": progress,
            "message": message,
            "status": status,
        })
        if analysis_id in self.active_connections:
            dead = []
            for ws in self.active_connections[analysis_id]:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(analysis_id, ws)


manager = ConnectionManager()


@router.websocket("/ws/analyses/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """WebSocket endpoint for streaming analysis progress."""
    await manager.connect(analysis_id, websocket)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Client can send ping/pong or commands
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(analysis_id, websocket)
        logger.info(f"WebSocket disconnected for analysis {analysis_id}")
