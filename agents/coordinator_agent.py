"""
CoordinatorAgent: Orchestrates the entire RAG workflow
"""

import sys
import os
from typing import Dict, Any, Optional
import asyncio
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.message_protocol import MCPMessage, MessageTypes, message_bus

class CoordinatorAgent:
    """Agent responsible for orchestrating the RAG workflow"""
    
    def __init__(self):
        self.name = "CoordinatorAgent"
        self.active_requests = {}  # Track ongoing requests
        self.session_results = {}  # Store results for UI
        
        # Subscribe to relevant messages
        message_bus.subscribe(self.name, self.handle_message)
        
        print(f"🎯 CoordinatorAgent: Initialized and ready to coordinate workflows")
    
    def handle_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.type == MessageTypes.CONTEXT_RESPONSE:
            self.handle_indexing_complete(message)
        elif message.type == MessageTypes.LLM_RESPONSE:
            self.handle_llm_response(message)
        elif message.type == MessageTypes.ERROR:
            self.handle_error(message)
    
    def process_document_upload(self, filename: str, file_content: bytes) -> str:
        """Initiate document processing workflow"""
        # Create ingestion request
        ingestion_message = message_bus.create_message(
            sender=self.name,
            receiver="IngestionAgent",
            msg_type=MessageTypes.INGESTION_REQUEST,
            payload={
                'filename': filename,
                'file_content': file_content
            }
        )
        
        trace_id = ingestion_message.trace_id
        
        # Track the request
        self.active_requests[trace_id] = {
            'type': 'document_upload',
            'filename': filename,
            'status': 'processing',
            'start_time': time.time()
        }
        
        print(f"🚀 CoordinatorAgent: Starting document upload workflow for '{filename}' [trace: {trace_id}]")
        message_bus.publish(ingestion_message)
        
        return trace_id
    
    def process_user_query(self, query: str, top_k: int = 5) -> str:
        """Initiate query processing workflow"""
        # Create retrieval request
        retrieval_message = message_bus.create_message(
            sender=self.name,
            receiver="RetrievalAgent",
            msg_type=MessageTypes.RETRIEVAL_REQUEST,
            payload={
                'query': query,
                'top_k': top_k
            }
        )
        
        trace_id = retrieval_message.trace_id
        
        # Track the request
        self.active_requests[trace_id] = {
            'type': 'user_query',
            'query': query,
            'status': 'processing',
            'start_time': time.time()
        }
        
        print(f"🔍 CoordinatorAgent: Starting query processing workflow for '{query}' [trace: {trace_id}]")
        message_bus.publish(retrieval_message)
        
        return trace_id
    
    def handle_indexing_complete(self, message: MCPMessage):
        """Handle document indexing completion"""
        trace_id = message.trace_id
        payload = message.payload
        
        if trace_id in self.active_requests:
            request_info = self.active_requests[trace_id]
            request_info['status'] = 'completed'
            request_info['result'] = {
                'filename': payload['filename'],
                'total_vectors': payload['total_vectors'],
                'success': True
            }
            
            # Store result for UI
            self.session_results[trace_id] = request_info
            
            print(f"✅ CoordinatorAgent: Document indexing completed for [trace: {trace_id}]")
    
    def handle_llm_response(self, message: MCPMessage):
        """Handle LLM response completion"""
        trace_id = message.trace_id
        payload = message.payload
        
        if trace_id in self.active_requests:
            request_info = self.active_requests[trace_id]
            request_info['status'] = 'completed'
            request_info['result'] = {
                'query': payload['query'],
                'answer': payload['answer'],
                'sources': payload['sources'],
                'context_used': payload['context_used'],
                'success': True
            }
            
            # Store result for UI
            self.session_results[trace_id] = request_info
            
            print(f"💬 CoordinatorAgent: Query response completed for [trace: {trace_id}]")
    
    def handle_error(self, message: MCPMessage):
        """Handle error messages"""
        trace_id = message.trace_id
        payload = message.payload
        
        if trace_id in self.active_requests:
            request_info = self.active_requests[trace_id]
            request_info['status'] = 'error'
            request_info['result'] = {
                'error': payload['error'],
                'success': False
            }
            
            # Store result for UI
            self.session_results[trace_id] = request_info
            
            print(f"❌ CoordinatorAgent: Error occurred for [trace: {trace_id}]: {payload['error']}")
    
    def get_request_status(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request"""
        return self.active_requests.get(trace_id)
    
    def get_result(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed request"""
        return self.session_results.get(trace_id)
    
    def wait_for_result(self, trace_id: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for a request to complete and return result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if trace_id in self.session_results:
                result = self.session_results[trace_id]
                if result['status'] in ['completed', 'error']:
                    return result
            time.sleep(0.1)
        
        # Timeout occurred
        return {
            'status': 'timeout',
            'result': {
                'error': f'Request timed out after {timeout} seconds',
                'success': False
            }
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        total_requests = len(self.active_requests) + len(self.session_results)
        completed_requests = len([r for r in self.session_results.values() if r['status'] == 'completed'])
        error_requests = len([r for r in self.session_results.values() if r['status'] == 'error'])
        
        return {
            'total_requests': total_requests,
            'completed_requests': completed_requests,
            'error_requests': error_requests,
            'active_requests': len([r for r in self.active_requests.values() if r['status'] == 'processing']),
            'message_history_count': len(message_bus.messages)
        }
    
    def clear_session(self):
        """Clear session data"""
        self.active_requests.clear()
        self.session_results.clear()
        print("🧹 CoordinatorAgent: Session cleared")