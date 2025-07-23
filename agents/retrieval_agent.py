"""
RetrievalAgent: Handles embedding generation and semantic retrieval
"""

import sys
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.message_protocol import MCPMessage, MessageTypes, message_bus

class RetrievalAgent:
    """Agent responsible for embedding generation and semantic retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.name = "RetrievalAgent"
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
        self.documents = []  # Store document chunks with metadata
        self.document_map = {}  # Map from index to document info
        
        # Subscribe to relevant messages
        message_bus.subscribe(self.name, self.handle_message)
        
        print(f"🚀 RetrievalAgent: Initialized with model '{model_name}'")
    
    def handle_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.type == MessageTypes.INGESTION_COMPLETE:
            self.index_document(message)
        elif message.type == MessageTypes.RETRIEVAL_REQUEST:
            self.retrieve_context(message)
    
    def index_document(self, message: MCPMessage):
        """Index document chunks into vector store"""
        try:
            payload = message.payload
            doc_id = payload['doc_id']
            filename = payload['filename']
            chunks = payload['chunks']
            trace_id = message.trace_id
            
            print(f"📚 RetrievalAgent: Indexing document '{filename}' with {len(chunks)} chunks [trace: {trace_id}]")
            
            # Generate embeddings for all chunks
            embeddings = self.embedding_model.encode(chunks)
            
            # Normalize embeddings for inner product similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            start_idx = len(self.documents)
            self.index.add(embeddings.astype(np.float32))
            
            # Store document metadata
            for i, chunk in enumerate(chunks):
                doc_info = {
                    'doc_id': doc_id,
                    'filename': filename,
                    'chunk_index': i,
                    'chunk_text': chunk,
                    'file_type': payload.get('file_type', 'unknown')
                }
                self.documents.append(doc_info)
                self.document_map[start_idx + i] = doc_info
            
            print(f"✅ RetrievalAgent: Successfully indexed '{filename}' -> Total vectors: {self.index.ntotal}")
            
            # Confirm indexing complete
            response_message = message_bus.create_message(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type=MessageTypes.CONTEXT_RESPONSE,
                payload={
                    'status': 'indexed',
                    'doc_id': doc_id,
                    'filename': filename,
                    'total_vectors': self.index.ntotal,
                    'trace_id': trace_id
                }
            )
            response_message.trace_id = trace_id
            message_bus.publish(response_message)
            
        except Exception as e:
            error_message = message_bus.create_message(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type=MessageTypes.ERROR,
                payload={
                    'error': f"Indexing failed: {str(e)}",
                    'trace_id': message.trace_id
                }
            )
            error_message.trace_id = message.trace_id
            print(f"💥 RetrievalAgent: Indexing error: {str(e)}")
            message_bus.publish(error_message)
    
    def retrieve_context(self, message: MCPMessage):
        """Retrieve relevant context for a query"""
        try:
            query = message.payload['query']
            top_k = message.payload.get('top_k', 5)
            trace_id = message.trace_id
            
            print(f"🔍 RetrievalAgent: Searching for '{query}' (top {top_k}) [trace: {trace_id}]")
            
            if self.index.ntotal == 0:
                # No documents indexed yet
                response_message = message_bus.create_message(
                    sender=self.name,
                    receiver="LLMResponseAgent",
                    msg_type=MessageTypes.RETRIEVAL_RESULT,
                    payload={
                        'query': query,
                        'retrieved_context': [],
                        'source_documents': [],
                        'trace_id': trace_id,
                        'message': 'No documents have been uploaded yet.'
                    }
                )
                response_message.trace_id = trace_id
                message_bus.publish(response_message)
                return
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search for similar chunks
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            # Prepare retrieved context
            retrieved_context = []
            source_documents = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score > 0.1:  # Filter low similarity scores
                    doc_info = self.document_map[idx]
                    retrieved_context.append(doc_info['chunk_text'])
                    source_documents.append({
                        'filename': doc_info['filename'],
                        'chunk_index': doc_info['chunk_index'],
                        'similarity_score': float(score),
                        'file_type': doc_info['file_type']
                    })
            
            print(f"📊 RetrievalAgent: Found {len(retrieved_context)} relevant chunks")
            
            # Send results to LLMResponseAgent
            response_message = message_bus.create_message(
                sender=self.name,
                receiver="LLMResponseAgent",
                msg_type=MessageTypes.RETRIEVAL_RESULT,
                payload={
                    'query': query,
                    'retrieved_context': retrieved_context,
                    'source_documents': source_documents,
                    'trace_id': trace_id
                }
            )
            response_message.trace_id = trace_id
            message_bus.publish(response_message)
            
        except Exception as e:
            error_message = message_bus.create_message(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type=MessageTypes.ERROR,
                payload={
                    'error': f"Retrieval failed: {str(e)}",
                    'query': message.payload.get('query', 'unknown'),
                    'trace_id': message.trace_id
                }
            )
            error_message.trace_id = message.trace_id
            print(f"💥 RetrievalAgent: Retrieval error: {str(e)}")
            message_bus.publish(error_message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval agent statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'total_documents': len(set(doc['doc_id'] for doc in self.documents)),
            'total_chunks': len(self.documents),
            'model_name': self.embedding_model.get_sentence_embedding_dimension()
        }