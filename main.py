"""
Phishing Campaign Analyzer & Knowledge Chatbot
Cloud-Native FastAPI Backend with Qdrant Cloud Integration
"""

import os
import logging
import uuid
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
import io
import asyncio
from datetime import datetime

# Import core modules
from data_processor import PhishingDataProcessor
from insight_generator import InsightGenerator
from embeddings import EmbeddingGenerator
from vector_store import VectorStore, RAGRetriever
from llm_orchestrator import LLMOrchestrator
from pdf_processor import PDFProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================
# Pydantic Models
# ============================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's natural language question")
    collection: Optional[str] = Field(None, description="Specific collection to query (phishing_insights, company_knowledge, phishing_general)")
    campaign_id: Optional[str] = Field(None, description="Specific campaign ID to query")
    include_sources: bool = Field(True, description="Include source citations in response")

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any]

class UploadResponse(BaseModel):
    status: str
    message: str
    collection: str
    chunks_added: int
    campaign_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]
    collections: List[str]
    message: Optional[str] = Field(None, description="Status message")
    environment: Optional[Dict[str, str]] = Field(None, description="Environment variable status")

class StreamQueryRequest(BaseModel):
    query: str = Field(..., description="User's natural language question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    collection: Optional[str] = Field(None, description="Specific collection to query")
    include_sources: bool = Field(True, description="Include source citations in response")

class FeedbackRequest(BaseModel):
    message_id: str = Field(..., description="ID of the message being rated")
    session_id: str = Field(..., description="Session ID")
    rating: str = Field(..., description="Rating: 'up' or 'down'")
    query: Optional[str] = Field(None, description="Original query")
    response: Optional[str] = Field(None, description="Bot response")

class FeedbackResponse(BaseModel):
    status: str
    message: str

# ============================================
# Global State
# ============================================

class AppState:
    def __init__(self):
        self.embedding_generator = None
        self.vector_store = None
        self.rag_retriever = None
        self.llm_orchestrator = None
        self.data_processor = None
        self.pdf_processor = None
        self.initialized = False
        # Session-based conversation memory
        self.sessions: Dict[str, Dict] = {}
        # Feedback storage
        self.feedback: List[Dict] = []

app_state = AppState()

# ============================================
# Session Management
# ============================================

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create a new one"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in app_state.sessions:
        app_state.sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "conversation_history": [],
            "message_count": 0
        }
    
    return session_id

def add_to_session(session_id: str, query: str, response: str, message_id: str):
    """Add a message exchange to session history"""
    if session_id in app_state.sessions:
        app_state.sessions[session_id]["conversation_history"].append({
            "message_id": message_id,
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        app_state.sessions[session_id]["message_count"] += 1
        
        # Keep only last 20 messages per session
        if len(app_state.sessions[session_id]["conversation_history"]) > 20:
            app_state.sessions[session_id]["conversation_history"] = \
                app_state.sessions[session_id]["conversation_history"][-20:]

# ============================================
# Startup & Shutdown
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup"""
    logger.info("🚀 Initializing Phishing Campaign Analyzer & Knowledge Chatbot...")
    
    try:
        # Get configuration from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")  # Optional for public models
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        qdrant_url = os.getenv("QDRANT_URL")  # Qdrant Cloud URL
        qdrant_api_key = os.getenv("QDRANT_API_KEY")  # Qdrant Cloud API Key
        
        if not groq_api_key:
            logger.warning("⚠️ GROQ_API_KEY not set - LLM features will be limited")
        
        if not huggingface_api_key:
            logger.info("ℹ️ HUGGINGFACE_API_KEY not set - using public model (no API key required)")
        
        # Initialize Embedding Generator
        logger.info(f"📊 Initializing embedding generator with model: {embedding_model}...")
        try:
            app_state.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
            app_state.embedding_generator.load_model()
            # Verify embedding generator is working by checking dimension
            if app_state.embedding_generator.dimension is None:
                # Default dimension for BAAI/bge-small-en-v1.5 is 384
                default_dim = 384
                logger.warning(f"⚠️ Could not determine embedding dimension - using default {default_dim}")
                app_state.embedding_generator.dimension = default_dim
            logger.info(f"✅ Embedding generator ready (dimension: {app_state.embedding_generator.dimension})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding generator: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error("💡 Check that the model name is correct and you have internet connection to download the model")
            logger.error(f"💡 Model: {embedding_model}")
            raise
        
        # Initialize PDF Processor
        logger.info("📄 Initializing PDF processor...")
        app_state.pdf_processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
        logger.info("✅ PDF processor ready")
        
        # Initialize Vector Store (Qdrant Cloud)
        logger.info("🗄️ Connecting to Qdrant Cloud...")
        # Get dimension from embedding generator if available, otherwise use default
        # Default for BAAI/bge-small-en-v1.5 is 384
        vector_size = app_state.embedding_generator.dimension or 384
        
        app_state.vector_store = VectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            vector_size=vector_size
        )
        await app_state.vector_store.connect()
        
        # Create collections for different knowledge domains
        # Use environment variables if set, otherwise use defaults
        collections = [
            os.getenv("COLLECTION_PHISHING", "phishing_insights"),      # Campaign analytics
            os.getenv("COLLECTION_COMPANY", "company_knowledge"),        # Organization info
            os.getenv("COLLECTION_GENERAL", "phishing_general"),         # General phishing knowledge
            "pdf_documents"          # PDF documents
        ]
        
        for collection in collections:
            await app_state.vector_store.create_collection(
                collection_name=collection,
                recreate=False  # Don't delete existing data
            )
            logger.info(f"✅ Collection '{collection}' ready")
        
        # Initialize RAG Retriever
        logger.info("🔍 Setting up RAG retriever...")
        app_state.rag_retriever = RAGRetriever(
            vector_store=app_state.vector_store,
            embedding_generator=app_state.embedding_generator
        )
        logger.info("✅ RAG retriever ready")
        
        # Initialize LLM Orchestrator
        if groq_api_key:
            logger.info("🤖 Connecting to Groq LLM...")
            try:
                groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                app_state.llm_orchestrator = LLMOrchestrator(
                    api_key=groq_api_key,
                    model=groq_model,
                    rag_retriever=app_state.rag_retriever
                )
                logger.info(f"✅ LLM orchestrator ready (model: {groq_model})")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LLM orchestrator: {e}")
                logger.error("💡 Check that GROQ_API_KEY is valid. Get one at https://console.groq.com")
                raise
        else:
            logger.warning("⚠️ GROQ_API_KEY not set - query endpoint will not work")
            logger.warning("💡 Get a free API key at https://console.groq.com")
        
        app_state.initialized = True
        logger.info("✨ System fully initialized and ready!")
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        logger.error(f"❌ Full error details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        # Don't raise - allow app to start but mark as not initialized
        # This way health check can still work
        app_state.initialized = False
        logger.error("⚠️ System started but initialization failed. Check logs above for details.")
        logger.error("💡 Visit /health endpoint to see which components failed to initialize")
        logger.error("💡 Common fixes:")
        logger.error("   - Verify GROQ_API_KEY is set and valid")
        logger.error("   - Verify MIXEDBREAD_API_KEY is set and valid")
        logger.error("   - Verify QDRANT_URL and QDRANT_API_KEY are set (or system will use in-memory)")
        logger.error("   - Check Railway logs for detailed error messages")
    
    yield
    
    # Cleanup
    logger.info("👋 Shutting down...")

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="Phishing Campaign Analyzer & Knowledge Chatbot",
    description="Cloud-native RAG system for phishing analytics and knowledge assistance",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Middleware - Configure from environment
# For web chatbot integration, we need to allow all origins by default
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_env == "*":
    allowed_origins = ["*"]
    logger.info("🌐 CORS: Allowing all origins (*)")
else:
    # Split comma-separated origins
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",")]
    logger.info(f"🌐 CORS: Allowing origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Phishing Campaign Analyzer & Knowledge Chatbot",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint - Use this to diagnose initialization issues"""
    components = {
        "embedding_generator": app_state.embedding_generator is not None,
        "vector_store": app_state.vector_store is not None,
        "rag_retriever": app_state.rag_retriever is not None,
        "llm_orchestrator": app_state.llm_orchestrator is not None,
    }
    
    # Check environment variables
    env_status = {
        "GROQ_API_KEY": "set" if os.getenv("GROQ_API_KEY") else "missing",
        "HUGGINGFACE_API_KEY": "set" if os.getenv("HUGGINGFACE_API_KEY") else "optional (public models)",
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "QDRANT_URL": "set" if os.getenv("QDRANT_URL") else "missing (using in-memory)",
        "QDRANT_API_KEY": "set" if os.getenv("QDRANT_API_KEY") else "missing (using in-memory)",
    }
    
    collections = []
    if app_state.vector_store:
        try:
            # Get all collections from Qdrant
            collections_info = await app_state.vector_store.client.get_collections()
            collections = [col.name for col in collections_info.collections]
        except Exception as e:
            logger.error(f"Error fetching collections: {e}")
    
    # Check if all required components exist (more reliable than initialized flag)
    all_components_ready = all(components.values())
    status = "healthy" if (app_state.initialized and all_components_ready) else "error"
    
    if app_state.initialized and all_components_ready:
        message = "System is ready"
    else:
        message = "System initialization failed - check logs"
        missing_components = [k for k, v in components.items() if not v]
        if missing_components:
            message += f". Missing components: {', '.join(missing_components)}"
        elif not app_state.initialized:
            message += f". Initialized flag is False but components exist: {[k for k, v in components.items() if v]}"
    
    return {
        "status": status,
        "components": components,
        "collections": collections,
        "message": message,
        "environment": env_status
    }

@app.get("/test", tags=["System"])
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {
        "status": "ok",
        "message": "API is responding",
        "initialized": app_state.initialized,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/query_agent", response_model=QueryResponse, tags=["Query"])
async def query_agent(request: QueryRequest):
    """
    Main query endpoint - Handles natural language questions
    Supports queries about:
    - Phishing campaigns (click rates, risk analysis)
    - Company knowledge (who we are, what we do)
    - General phishing information (tactics, defenses)
    
    Perfect for chatbot integration - returns JSON with response and sources.
    """
    # Check if all required components exist (more reliable check)
    # Check all required components
    if not app_state.llm_orchestrator:
        logger.error("❌ LLM orchestrator not available")
        raise HTTPException(
            status_code=503, 
            detail="LLM not available. Please set GROQ_API_KEY environment variable. Get a free API key at https://console.groq.com"
        )
    
    if not app_state.embedding_generator:
        logger.error("❌ Embedding generator not available")
        raise HTTPException(
            status_code=503,
            detail="Embedding generator not available. Check server logs for initialization errors."
        )
    
    if not app_state.vector_store:
        logger.error("❌ Vector store not available")
        raise HTTPException(
            status_code=503,
            detail="Vector store not available. Check QDRANT_URL and QDRANT_API_KEY or check server logs."
        )
    
    if not app_state.rag_retriever:
        logger.error("❌ RAG retriever not available")
        raise HTTPException(
            status_code=503,
            detail="RAG retriever not available. This is an internal error - check server logs."
        )
    
    try:
        logger.info(f"📥 Query received: {request.query[:100]}...")
        
        # Process query through orchestrator
        response_text, sources = await app_state.llm_orchestrator.process_query(
            query=request.query,
            collection=request.collection,
            include_sources=request.include_sources
        )
        
        # Build metadata
        metadata = {
            "query_length": len(request.query),
            "response_length": len(response_text),
            "sources_count": len(sources) if sources else 0,
            "collection_used": request.collection or "auto-detected"
        }
        
        logger.info(f"✅ Query processed successfully")
        
        return QueryResponse(
            query=request.query,
            response=response_text,
            sources=sources if request.include_sources else None,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Streaming Query Endpoint
# ============================================

@app.post("/query_stream", tags=["Query"])
async def query_stream(request: StreamQueryRequest):
    """
    Streaming query endpoint - Returns response in real-time chunks via SSE.
    Much better UX as users see the response being generated.
    """
    # Check all required components
    if not app_state.llm_orchestrator:
        raise HTTPException(status_code=503, detail="LLM not available")
    if not app_state.embedding_generator:
        raise HTTPException(status_code=503, detail="Embedding generator not available")
    if not app_state.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not available")
    if not app_state.rag_retriever:
        raise HTTPException(status_code=503, detail="RAG retriever not available")
    
    # Get or create session
    session_id = get_or_create_session(request.session_id)
    message_id = str(uuid.uuid4())
    
    async def generate_stream():
        """Generator for SSE streaming"""
        full_response = ""
        sources = None
        suggested_questions = []
        
        try:
            # Send session info first
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'message_id': message_id})}\n\n"
            
            # Send typing indicator
            yield f"data: {json.dumps({'type': 'typing', 'status': 'start'})}\n\n"
            
            # Get context (use smaller top_k for faster responses)
            rag_top_k = int(os.getenv('RAG_TOP_K', '3'))
            contexts = await app_state.rag_retriever.retrieve_context(
                query=request.query,
                collection=request.collection,
                top_k=rag_top_k
            )
            
            # Format context
            formatted_context = app_state.rag_retriever.format_context_for_llm(contexts) if contexts else ""
            
            # Prepare sources
            if request.include_sources and contexts:
                sources = [
                    {
                        'content': ctx['payload'].get('text', '')[:200] + '...',
                        'relevance': ctx['score'],
                        'collection': ctx.get('collection', 'unknown'),
                        'title': ctx['payload'].get('title', ctx['payload'].get('campaign_name', 'Untitled'))
                    }
                    for ctx in contexts[:5]
                ]
            
            # Stream the response
            async for chunk in app_state.llm_orchestrator.stream_response(
                query=request.query,
                context=formatted_context,
                session_id=session_id
            ):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
            # Save to session
            add_to_session(session_id, request.query, full_response, message_id)
            
            # Send completion with sources (skip follow-up questions for speed)
            yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'suggested_questions': [], 'message_id': message_id})}\n\n"
            
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================
# Feedback Endpoint
# ============================================

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a bot response (thumbs up/down).
    Helps improve response quality over time.
    """
    try:
        feedback_entry = {
            "id": str(uuid.uuid4()),
            "message_id": request.message_id,
            "session_id": request.session_id,
            "rating": request.rating,
            "query": request.query,
            "response": request.response,
            "timestamp": datetime.now().isoformat()
        }
        
        app_state.feedback.append(feedback_entry)
        
        # Keep only last 1000 feedback entries in memory
        if len(app_state.feedback) > 1000:
            app_state.feedback = app_state.feedback[-1000:]
        
        logger.info(f"📝 Feedback received: {request.rating} for message {request.message_id}")
        
        return FeedbackResponse(
            status="success",
            message=f"Thank you for your feedback!"
        )
        
    except Exception as e:
        logger.error(f"❌ Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/stats", tags=["Feedback"])
async def get_feedback_stats():
    """Get feedback statistics"""
    total = len(app_state.feedback)
    up_votes = sum(1 for f in app_state.feedback if f["rating"] == "up")
    down_votes = sum(1 for f in app_state.feedback if f["rating"] == "down")
    
    return {
        "total_feedback": total,
        "up_votes": up_votes,
        "down_votes": down_votes,
        "satisfaction_rate": round(up_votes / total * 100, 1) if total > 0 else 0
    }

# ============================================
# Quick Actions Endpoint
# ============================================

@app.get("/quick_actions", tags=["Query"])
async def get_quick_actions():
    """
    Get pre-defined quick action buttons for common queries.
    These appear in the chat UI for easy access.
    """
    return {
        "quick_actions": [
            {
                "id": "department_summary",
                "label": "📊 Department Summary",
                "query": "Give me a summary of click rates by department",
                "icon": "📊"
            },
            {
                "id": "risky_users",
                "label": "⚠️ Risky Users",
                "query": "Who are the top 5 riskiest users and why?",
                "icon": "⚠️"
            },
            {
                "id": "phishing_tactics",
                "label": "🧠 Phishing Tactics",
                "query": "What are the most common phishing tactics?",
                "icon": "🧠"
            },
            {
                "id": "defense_tips",
                "label": "🛡️ Defense Tips",
                "query": "How can we protect against phishing attacks?",
                "icon": "🛡️"
            },
            {
                "id": "campaign_overview",
                "label": "📈 Campaign Overview",
                "query": "Give me an overview of our phishing campaigns",
                "icon": "📈"
            },
            {
                "id": "about_company",
                "label": "🏢 About Us",
                "query": "Tell me about our company",
                "icon": "🏢"
            }
        ]
    }

# ============================================
# Session Management Endpoints
# ============================================

@app.get("/session/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    """Get session information and conversation history"""
    if session_id not in app_state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        **app_state.sessions[session_id]
    }

@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """Clear a session's conversation history"""
    if session_id in app_state.sessions:
        app_state.sessions[session_id]["conversation_history"] = []
        app_state.sessions[session_id]["message_count"] = 0
        
        # Also clear LLM orchestrator history
        if app_state.llm_orchestrator:
            app_state.llm_orchestrator.clear_history()
    
    return {"status": "success", "message": "Session cleared"}

@app.post("/upload/phishing_campaign", response_model=UploadResponse, tags=["Upload"])
async def upload_phishing_campaign(
    file: UploadFile = File(...),
    campaign_name: str = Form(...),
    campaign_description: Optional[str] = Form(None)
):
    """
    Upload a phishing campaign CSV file
    Automatically generates insights and stores in vector database
    """
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        logger.info(f"📤 Uploading campaign: {campaign_name}")
        
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Save to temporary file for processing
        temp_path = f"temp_{campaign_name}.csv"
        df.to_csv(temp_path, index=False)
        
        # Process with PhishingDataProcessor
        processor = PhishingDataProcessor(temp_path)
        # processor.load_data() - Now called in __init__
        logger.info(f"✅ Loaded {len(processor.df)} records")
        
        # Generate insights
        insight_generator = InsightGenerator(processor)
        insights = insight_generator.generate_all_insights()
        
        # Add campaign metadata
        campaign_id = f"campaign_{campaign_name.lower().replace(' ', '_')}"
        for insight in insights:
            insight['campaign_id'] = campaign_id
            insight['campaign_name'] = campaign_name
            if campaign_description:
                insight['campaign_description'] = campaign_description
        
        # Create embeddings
        insights_with_embeddings = app_state.embedding_generator.encode_insights(insights)
        
        # Store in Qdrant Cloud (phishing_insights collection)
        await app_state.vector_store.add_documents(
            documents=insights_with_embeddings,
            collection_name="phishing_insights"
        )
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        logger.info(f"✅ Campaign uploaded: {len(insights)} insights added")
        
        return UploadResponse(
            status="success",
            message=f"Campaign '{campaign_name}' processed successfully",
            collection="phishing_insights",
            chunks_added=len(insights),
            campaign_id=campaign_id
        )
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/company_knowledge", response_model=UploadResponse, tags=["Upload"])
async def upload_company_knowledge(
    content: str = Form(...),
    title: str = Form(...),
    category: str = Form("general")
):
    """
    Upload company knowledge documents
    Examples: "Who we are", "What we do", "Our mission"
    """
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        logger.info(f"📤 Uploading company knowledge: {title}")
        
        # Split content into chunks (simple paragraph splitting)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        documents = []
        for i, para in enumerate(paragraphs):
            doc = {
                'id': f"{title.lower().replace(' ', '_')}_{i}",
                'text': para,
                'title': title,
                'category': category,
                'source_type': 'company_knowledge'
            }
            documents.append(doc)
        
        # Create embeddings
        texts = [doc['text'] for doc in documents]
        embeddings = app_state.embedding_generator.encode_text(texts)
        
        for doc, emb in zip(documents, embeddings):
            doc['embedding'] = emb.tolist()
        
        # Store in Qdrant Cloud (company_knowledge collection)
        await app_state.vector_store.add_documents(
            documents=documents,
            collection_name="company_knowledge"
        )
        
        logger.info(f"✅ Company knowledge uploaded: {len(documents)} chunks")
        
        return UploadResponse(
            status="success",
            message=f"Company knowledge '{title}' added successfully",
            collection="company_knowledge",
            chunks_added=len(documents)
        )
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/phishing_general", response_model=UploadResponse, tags=["Upload"])
async def upload_phishing_general(
    content: str = Form(...),
    title: str = Form(...),
    topic: str = Form("general")
):
    """
    Upload general phishing knowledge
    Examples: "What is phishing?", "Common phishing tactics", "Defense strategies"
    """
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        logger.info(f"📤 Uploading phishing knowledge: {title}")
        
        # Split content into chunks
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        documents = []
        for i, para in enumerate(paragraphs):
            doc = {
                'id': f"{title.lower().replace(' ', '_')}_{i}",
                'text': para,
                'title': title,
                'topic': topic,
                'source_type': 'phishing_general'
            }
            documents.append(doc)
        
        # Create embeddings
        texts = [doc['text'] for doc in documents]
        embeddings = app_state.embedding_generator.encode_text(texts)
        
        for doc, emb in zip(documents, embeddings):
            doc['embedding'] = emb.tolist()
        
        # Store in Qdrant Cloud (phishing_general collection)
        await app_state.vector_store.add_documents(
            documents=documents,
            collection_name="phishing_general"
        )
        
        logger.info(f"✅ Phishing knowledge uploaded: {len(documents)} chunks")
        
        return UploadResponse(
            status="success",
            message=f"Phishing knowledge '{title}' added successfully",
            collection="phishing_general",
            chunks_added=len(documents)
        )
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/pdf_document", response_model=UploadResponse, tags=["Upload"])
async def upload_pdf_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form("general"),
    description: Optional[str] = Form(None)
):
    """
    Upload a PDF document
    Extracts text, chunks it, and stores in vector database for querying
    """
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not app_state.pdf_processor:
        raise HTTPException(status_code=503, detail="PDF processor not initialized")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        logger.info(f"📤 Uploading PDF document: {title}")
        
        # Read PDF content
        pdf_bytes = await file.read()
        
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="PDF file is empty")
        
        # Process PDF: extract text and chunk it
        metadata = {
            'filename': file.filename,
            'file_size': len(pdf_bytes)
        }
        if description:
            metadata['description'] = description
        
        chunks = app_state.pdf_processor.process_pdf(
            pdf_bytes=pdf_bytes,
            title=title,
            category=category,
            metadata=metadata
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # Create embeddings for chunks
        texts = [chunk['text'] for chunk in chunks]
        embeddings = app_state.embedding_generator.encode_text(texts)
        
        # Add embeddings to chunks
        for chunk, emb in zip(chunks, embeddings):
            chunk['embedding'] = emb.tolist()
        
        # Store in Qdrant Cloud (pdf_documents collection)
        await app_state.vector_store.add_documents(
            documents=chunks,
            collection_name="pdf_documents"
        )
        
        logger.info(f"✅ PDF document uploaded: {len(chunks)} chunks added")
        
        return UploadResponse(
            status="success",
            message=f"PDF document '{title}' processed successfully",
            collection="pdf_documents",
            chunks_added=len(chunks)
        )
        
    except ValueError as e:
        logger.error(f"❌ PDF processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/info", tags=["Collections"])
async def get_collection_info(collection_name: str):
    """Get information about a specific collection"""
    if not app_state.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        info = await app_state.vector_store.get_collection_info(collection_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

@app.get("/campaigns/list", tags=["Campaigns"])
async def list_campaigns():
    """List all available campaigns"""
    if not app_state.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        # Query all documents from phishing_insights collection
        # This is a simplified version - in production, you'd want pagination
        results = await app_state.vector_store.get_all_documents(
            collection_name="phishing_insights",
            limit=1000
        )
        
        # Extract unique campaign IDs
        campaigns = {}
        for result in results:
            campaign_id = result.get('campaign_id')
            if campaign_id and campaign_id not in campaigns:
                campaigns[campaign_id] = {
                    'campaign_id': campaign_id,
                    'campaign_name': result.get('campaign_name', 'Unknown'),
                    'campaign_description': result.get('campaign_description', '')
                }
        
        return {
            "total_campaigns": len(campaigns),
            "campaigns": list(campaigns.values())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    # Only enable reload in development
    reload = (environment == "development")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )