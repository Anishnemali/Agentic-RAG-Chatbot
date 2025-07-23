# 🚀 Agentic RAG Chatbot with Groq

A sophisticated multi-agent Retrieval-Augmented Generation (RAG) chatbot system built with Streamlit, powered by Groq's Llama3 model via LangChain. This system uses a distributed agent architecture with Model Context Protocol (MCP) for seamless document processing and intelligent Q&A.

## ✨ Features

- **Multi-Format Document Support**: PDF, DOCX, PPTX, CSV, TXT, Markdown
- **Intelligent Document Processing**: Automatic parsing, chunking, and vectorization
- **Semantic Search**: FAISS-powered vector similarity search with sentence transformers
- **Advanced LLM Integration**: Groq's Llama3-8B model with LangChain
- **Multi-Agent Architecture**: Distributed system with specialized agents
- **Real-time Chat Interface**: Streamlit-based interactive UI
- **Source Attribution**: Automatic citation of relevant document chunks
- **Fallback Processing**: Works without API keys using template-based responses

## 🏗️ Architecture

The system uses a multi-agent architecture with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ CoordinatorAgent│    │ IngestionAgent  │    │ RetrievalAgent  │
│                 │    │                 │    │                 │
│ • Orchestrates  │◄──►│ • Document      │◄──►│ • Embedding     │
│   workflows     │    │   parsing       │    │   generation    │
│ • Tracks        │    │ • Text chunking │    │ • FAISS indexing│
│   requests      │    │ • Preprocessing │    │ • Similarity    │
│                 │    │                 │    │   search        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                              │
         │                                              ▼
┌─────────────────┐           MCP Bus            ┌─────────────────┐
│ LLMResponseAgent│◄─────────────────────────────►│ Streamlit UI    │
│                 │                               │                 │
│ • Groq API      │                               │ • File upload   │
│   integration   │                               │ • Chat interface│
│ • Response      │                               │ • Results       │
│   generation    │                               │   display       │
└─────────────────┘                               └─────────────────┘
```
Flow Diagram
```
                        ┌───────────────────────────────┐
                        │        User Uploads File      │
                        └──────────────┬────────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │   Streamlit Frontend   │
                          │- Uploads files         │
                          │- Chat interface        │
                          └──────────┬─────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────────┐
                     │         CoordinatorAgent           │
                     │- Orchestrates pipeline             │
                     │- Routes messages to other agents   │
                     └────┬────────────┬───────────────┬──┘
                          │            │               │
          ┌───────────────▼─┐      ┌───▼──────────┐ ┌───▼────────────┐
          │ IngestionAgent  │      │ RetrievalAgent│ │LLMResponseAgent│
          │ - File parsing  │      │ - FAISS index │ │- Query Groq API│
          │ - Text chunking │      │ - Similarity  │ │- Generate reply│
          └────────────┬────┘      │   search      │ └───────┬────────┘
                       │           └──────┬─────────┘         │
                       ▼                  ▼                   │
            ┌────────────────┐    ┌────────────────────┐      │
            │ DocumentParser │    │ SentenceTransformer│      │
            │ (PDF, DOCX, etc)│    │ + FAISS Vector DB │      │
            └────────────────┘    └────────────────────┘      │
                                                            ▼
                                          ┌────────────────────────────┐
                                          │   Response Sent to Frontend│
                                          └───────────────┬────────────┘
                                                          ▼
                                              ┌────────────────────┐
                                              │    Streamlit UI    │
                                              │  (Answer Displayed)│
                                              └────────────────────┘
```

### Agent Responsibilities

- **CoordinatorAgent**: Orchestrates workflows, tracks requests, manages session state
- **IngestionAgent**: Parses documents, handles text chunking and preprocessing  
- **RetrievalAgent**: Generates embeddings, manages FAISS vector store, performs similarity search
- **LLMResponseAgent**: Integrates with Groq API, generates contextual responses
- **MCP Bus**: Facilitates structured message passing between agents

## 📋 Requirements

### System Dependencies
```
Python 3.8+
```

### Python Packages
```
streamlit==1.28.0
langchain==0.1.0
langchain-groq==0.0.5
faiss-cpu==1.7.4
sentence-transformers==2.2.2
pypdf2==3.0.1
python-docx==1.1.0
python-pptx==0.6.23
pandas==2.1.0
markdown==3.5.1
python-dotenv==1.0.0
pdfplumber==0.9.0
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd agentic-rag-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

To get a free Groq API key:
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Generate an API key
4. Add it to your `.env` file

### 4. Run the Application
```bash
streamlit run app.py
```

### 5. Upload Documents and Start Chatting
1. Use the sidebar to upload documents (PDF, DOCX, PPTX, CSV, TXT, MD)
2. Wait for processing to complete
3. Ask questions about your documents in the chat interface

## 🔧 Configuration

### Model Settings
The system uses Groq's `llama3-8b-8192` model by default. You can modify this in `llm_response_agent.py`:

```python
llm_response = LLMResponseAgent(model_name="llama3-8b-8192")
```

### Chunking Parameters
Adjust text chunking in `document_parsers.py`:

```python
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    # Modify chunk_size and overlap as needed
```

### Retrieval Settings
Configure similarity search in `retrieval_agent.py`:

```python
# Number of similar chunks to retrieve
top_k = 5

# Minimum similarity threshold
score_threshold = 0.1
```

## 📁 Project Structure

```
agentic-rag-chatbot/
├── agents/
│   ├── __init__.py
│   ├── coordinator_agent.py      # Workflow orchestration
│   ├── ingestion_agent.py        # Document processing
│   ├── retrieval_agent.py        # Vector search
│   └── llm_response_agent.py     # LLM integration
├── mcp/
│   └── message_protocol.py       # Inter-agent communication
├── parsers/
│   └── document_parsers.py       # Multi-format document parsing
├── app.py                        # Streamlit UI
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## 🔌 API Integration

### Groq Integration
The system integrates with Groq's API using LangChain:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name="llama3-8b-8192",
    temperature=0.7,
    max_tokens=1000
)
```

### Fallback Mode
Without a Groq API key, the system provides template-based responses using the retrieved context, ensuring functionality even without external API access.

## 💡 Usage Examples

### Document Upload
```python
# Supported formats
files = ['document.pdf', 'slides.pptx', 'data.csv', 'notes.txt', 'guide.md']
```

### Sample Queries
- "What are the main topics covered in the uploaded documents?"
- "Summarize the key findings from the research paper"
- "What data trends are shown in the CSV file?"
- "Explain the methodology mentioned in the document"

## 🛠️ Advanced Features

### Message Tracing
Every request gets a unique trace ID for debugging:
```python
trace_id = coordinator.process_user_query("your question")
result = coordinator.wait_for_result(trace_id)
```

### System Statistics
Real-time monitoring of:
- Total documents processed
- Vector count in FAISS index
- Request success/failure rates
- Message history

### Session Management
- Clear session data
- Reset vector store
- Restart agent system

## 🐛 Troubleshooting

### Common Issues

**1. "No Groq API key found"**
- Ensure `.env` file exists in the root directory
- Verify `GROQ_API_KEY` is correctly set

**2. "Document parsing failed"**
- Check file format is supported
- Ensure file is not corrupted
- Try with a smaller file first

**3. "No relevant context found"**
- Upload more documents
- Try different query phrasing
- Check if documents contain relevant information

**4. FAISS/Embedding Issues**
```bash
pip install faiss-cpu --force-reinstall
pip install sentence-transformers --upgrade
```

### Debug Mode
Enable debug information in the UI:
- Expand "Advanced Settings & Debug"
- View MCP message history
- Check system statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Groq** for providing fast LLM inference
- **LangChain** for LLM integration framework
- **Streamlit** for the interactive web interface
- **FAISS** for efficient similarity search
- **Sentence Transformers** for semantic embeddings

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed description
4. Include system information and error logs

---

⭐ **Star this repository if you find it helpful!**
