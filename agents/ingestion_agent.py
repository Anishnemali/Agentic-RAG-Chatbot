"""
IngestionAgent: Handles document parsing and preprocessing
"""

from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.message_protocol import MCPMessage, MessageTypes, message_bus
from parsers.document_parsers import parse_document

class IngestionAgent:
    """Agent responsible for document parsing and preprocessing"""
    
    def __init__(self):
        self.name = "IngestionAgent"
        self.processed_documents = {}
        
        # Subscribe to relevant messages
        message_bus.subscribe(self.name, self.handle_message)
    
    def handle_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.type == MessageTypes.INGESTION_REQUEST:
            self.process_document(message)
    
    def process_document(self, message: MCPMessage):
        """Process uploaded document"""
        try:
            filename = message.payload.get('filename')
            file_content = message.payload.get('file_content')
            trace_id = message.trace_id
            
            print(f"🔄 IngestionAgent: Processing document '{filename}' [trace: {trace_id}]")
            
            # Parse the document
            result = parse_document(filename, file_content)
            
            if result['success']:
                # Store processed document
                doc_id = f"{filename}_{trace_id}"
                self.processed_documents[doc_id] = result
                
                # Send success response to RetrievalAgent
                response_message = message_bus.create_message(
                    sender=self.name,
                    receiver="RetrievalAgent",
                    msg_type=MessageTypes.INGESTION_COMPLETE,
                    payload={
                        'doc_id': doc_id,
                        'filename': filename,
                        'chunks': result['chunks'],
                        'chunk_count': result['chunk_count'],
                        'file_type': result['file_type'],
                        'trace_id': trace_id,
                        'success': True
                    }
                )
                response_message.trace_id = trace_id
                
                print(f"✅ IngestionAgent: Successfully processed '{filename}' -> {result['chunk_count']} chunks")
                message_bus.publish(response_message)
                
            else:
                # Send error response
                error_message = message_bus.create_message(
                    sender=self.name,
                    receiver="CoordinatorAgent",
                    msg_type=MessageTypes.ERROR,
                    payload={
                        'error': result['error'],
                        'filename': filename,
                        'trace_id': trace_id,
                        'success': False
                    }
                )
                error_message.trace_id = trace_id
                
                print(f"❌ IngestionAgent: Failed to process '{filename}': {result['error']}")
                message_bus.publish(error_message)
                
        except Exception as e:
            error_message = message_bus.create_message(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type=MessageTypes.ERROR,
                payload={
                    'error': str(e),
                    'filename': message.payload.get('filename', 'unknown'),
                    'trace_id': message.trace_id,
                    'success': False
                }
            )
            error_message.trace_id = message.trace_id
            
            print(f"💥 IngestionAgent: Exception processing document: {str(e)}")
            message_bus.publish(error_message)
    
    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Get information about a processed document"""
        return self.processed_documents.get(doc_id, {})
    
    def list_processed_documents(self) -> Dict[str, Any]:
        """List all processed documents"""
        return {doc_id: {
            'filename': info['filename'],
            'file_type': info['file_type'],
            'chunk_count': info['chunk_count']
        } for doc_id, info in self.processed_documents.items()}