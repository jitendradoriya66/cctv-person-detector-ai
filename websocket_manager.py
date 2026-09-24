import asyncio
import json
from typing import List, Dict, Any, Optional
from fastapi import WebSocket

class ConnectionManager:
    """
    Manages real-time bi-directional WebSocket connections between
    FastAPI backend and Jinja2 Web Dashboard browser clients.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔌 WebSocket client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"🔌 WebSocket client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Asynchronously broadcasts a JSON message to all connected WebSocket clients."""
        if not self.active_connections:
            return
            
        disconnected = []
        payload = json.dumps(message)
        
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception as e:
                print(f"⚠️ Error sending WebSocket message: {e}")
                disconnected.append(connection)
                
        for conn in disconnected:
            self.disconnect(conn)

    def sync_broadcast(self, message: Dict[str, Any]):
        """Thread-safe method to schedule WebSocket broadcast from background threads."""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

# Global Instance
ws_manager = ConnectionManager()
