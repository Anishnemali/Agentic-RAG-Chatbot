"""
LLMResponseAgent: Forms final LLM query and generates responses using Groq
"""

import sys
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.message_protocol import MCPMessage, MessageTypes, message_bus

class LLMResponseAgent:
    """Agent responsible for generating final responses using retrieved context with Groq"""
    
    def __init__(self, model_name: str = "llama3-8b-8192"):
        self.name = "LLMResponseAgent"
        self.model_name = model_name
        
        # Get Groq API key from environment
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        
        # Initialize Groq client
        self.use_groq = bool(self.groq_api_key)
        if self.use_groq:
            try:
                self.llm = ChatGroq(
                    groq_api_key=self.groq_api_key,
                    model_name=self.model_name,
                    temperature=0.7,
                    max_tokens=1000
                )
                print(f"🤖 LLMResponseAgent: Initialized with Groq model '{model_name}'")
            except Exception as e:
                print(f"⚠️ LLMResponseAgent: Failed to initialize Groq client: {str(e)}")
                self.use_groq = False
        
        if not self.use_groq:
            print(f"🤖 LLMResponseAgent: Initialized with fallback responses (no Groq API key or initialization failed)")
        
        # Subscribe to relevant messages
        message_bus.subscribe(self.name, self.handle_message)
    
    def handle_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.type == MessageTypes.RETRIEVAL_RESULT:
            self.generate_response(message)
    
    def generate_response(self, message: MCPMessage):
        """Generate response using retrieved context"""
        try:
            payload = message.payload
            query = payload['query']
            retrieved_context = payload['retrieved_context']
            source_documents = payload.get('source_documents', [])
            trace_id = message.trace_id
            
            print(f"💭 LLMResponseAgent: Generating response for '{query}' with {len(retrieved_context)} context chunks [trace: {trace_id}]")
            
            if not retrieved_context:
                # No relevant context found
                response = {
                    'answer': "I couldn't find relevant information in the uploaded documents to answer your question. Please make sure you've uploaded documents that contain information related to your query.",
                    'sources': [],
                    'context_used': False
                }
            else:
                # Generate response with context
                if self.use_groq:
                    response = self._generate_groq_response(query, retrieved_context, source_documents)
                else:
                    response = self._generate_fallback_response(query, retrieved_context, source_documents)
            
            # Send response back to coordinator
            response_message = message_bus.create_message(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type=MessageTypes.LLM_RESPONSE,
                payload={
                    'query': query,
                    'answer': response['answer'],
                    'sources': response['sources'],
                    'context_used': response['context_used'],
                    'trace_id': trace_id
                }
            )
            response_message.trace_id = trace_id
            
            print(f"✅ LLMResponseAgent: Response generated successfully")
            message_bus.publish(response_message)
            
        except Exception as e:
            error_message = message_bus.create_message(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type=MessageTypes.ERROR,
                payload={
                    'error': f"Response generation failed: {str(e)}",
                    'query': message.payload.get('query', 'unknown'),
                    'trace_id': message.trace_id
                }
            )
            error_message.trace_id = message.trace_id
            print(f"💥 LLMResponseAgent: Error generating response: {str(e)}")
            message_bus.publish(error_message)
    
    def _generate_groq_response(self, query: str, context: List[str], sources: List[Dict]) -> Dict[str, Any]:
        """Generate response using Groq API with LangChain"""
        try:
            # Prepare context for prompt
            context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context)])
            
            # Create system and user messages
            system_message = SystemMessage(content="""You are a helpful and knowledgeable assistant that answers questions based on provided context from uploaded documents. 

Your responsibilities:
1. Provide comprehensive, accurate answers based on the given context
2. If the context doesn't fully answer the question, acknowledge this limitation
3. Be specific and cite relevant parts of the context when possible
4. Maintain a professional yet conversational tone
5. Structure your response clearly with proper formatting when needed
6. If you find conflicting information in the context, mention it
7. Stay focused on the question asked and avoid unnecessary tangents

Always prioritize accuracy over completeness - it's better to provide a partial but correct answer than to speculate beyond the given context.""")
            
            user_prompt = f"""Based on the following context from uploaded documents, please answer the user's question comprehensively.

Context from documents:
{context_text}

User Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to fully answer the question, acknowledge this and provide what information you can find from the available context."""
            
            user_message = HumanMessage(content=user_prompt)
            
            # Call Groq API through LangChain
            response = self.llm.invoke([system_message, user_message])
            
            answer = response.content.strip()
            
            # Format sources
            formatted_sources = []
            for i, source in enumerate(sources):
                formatted_sources.append({
                    'filename': source['filename'],
                    'chunk_index': source['chunk_index'],
                    'similarity_score': round(source['similarity_score'], 3),
                    'file_type': source['file_type']
                })
            
            return {
                'answer': answer,
                'sources': formatted_sources,
                'context_used': True
            }
            
        except Exception as e:
            print(f"Groq API error: {str(e)}")
            return self._generate_fallback_response(query, context, sources)
    
    def _generate_fallback_response(self, query: str, context: List[str], sources: List[Dict]) -> Dict[str, Any]:
        """Generate enhanced fallback response without Groq API"""
        
        # More sophisticated template-based response
        if len(context) == 1:
            answer = f"""Based on the uploaded document, I found relevant information for your query "{query}":

**Key Information:**
{context[0][:600]}{"..." if len(context[0]) > 600 else ""}

*Note: This response was generated using fallback processing. For more sophisticated analysis and better answers, please configure your Groq API key in the .env file.*"""
        else:
            answer = f"""Based on the uploaded documents, I found {len(context)} relevant sections for your query "{query}":

"""
            # Show top 3 chunks with better formatting
            for i, chunk in enumerate(context[:3]):
                answer += f"""**Section {i+1}:**
{chunk[:400]}{"..." if len(chunk) > 400 else ""}

"""
            
            answer += "*Note: This response was generated using fallback processing. For more comprehensive analysis and better answers, please configure your Groq API key in the .env file.*"
        
        # Format sources
        formatted_sources = []
        for source in sources:
            formatted_sources.append({
                'filename': source['filename'],
                'chunk_index': source['chunk_index'],
                'similarity_score': round(source['similarity_score'], 3),
                'file_type': source['file_type']
            })
        
        return {
            'answer': answer,
            'sources': formatted_sources,
            'context_used': True
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            'model_name': self.model_name,
            'provider': 'Groq' if self.use_groq else 'Fallback',
            'api_key_configured': bool(self.groq_api_key),
            'status': 'active' if self.use_groq else 'fallback'
        }