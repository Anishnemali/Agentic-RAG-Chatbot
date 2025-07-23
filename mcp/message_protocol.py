"""
Model Context Protocol (MCP) Implementation
Handles structured message passing between agents
"""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import json

@dataclass
class MCPMessage:
    """Standard MCP Message Format"""
    sender: str
    receiver: str
    type: str
    trace_id: str
    payload: Dict[str, Any]
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "trace_id": self.trace_id,
            "payload": self.payload,
            "timestamp": self.timestamp
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

class MessageTypes:
    """Standard MCP Message Types"""
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    INGESTION_REQUEST = "INGESTION_REQUEST"
    INGESTION_COMPLETE = "INGESTION_COMPLETE"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESULT = "RETRIEVAL_RESULT"
    CONTEXT_RESPONSE = "CONTEXT_RESPONSE"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    ERROR = "ERROR"

class MCPBus:
    """In-memory message bus for agent communication"""
    
    def __init__(self):
        self.messages = []
        self.subscribers = {}
    
    def subscribe(self, agent_name: str, callback):
        """Subscribe an agent to receive messages"""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)
    
    def publish(self, message: MCPMessage):
        """Publish a message to the bus"""
        self.messages.append(message)
        
        # Notify receiver
        if message.receiver in self.subscribers:
            for callback in self.subscribers[message.receiver]:
                try:
                    callback(message)
                except Exception as e:
                    print(f"Error delivering message to {message.receiver}: {str(e)}")
    
    def create_message(self, sender: str, receiver: str, msg_type: str, payload: Dict[str, Any]) -> MCPMessage:
        """Create a new MCP message with auto-generated trace_id"""
        trace_id = str(uuid.uuid4())[:8]
        return MCPMessage(
            sender=sender,
            receiver=receiver,
            type=msg_type,
            trace_id=trace_id,
            payload=payload
        )
    
    def get_message_history(self, trace_id: Optional[str] = None) -> list:
        """Get message history, optionally filtered by trace_id"""
        if trace_id:
            return [msg for msg in self.messages if msg.trace_id == trace_id]
        return self.messages

# Global message bus instance
message_bus = MCPBus()