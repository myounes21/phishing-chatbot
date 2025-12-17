"""
Phishing Campaign Analyzer & Knowledge Chatbot
Cloud-Native FastAPI Backend with Qdrant Cloud Integration
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
import io

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
        self.initialized = False

app_state = AppState()

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
        qdrant_url = os.getenv("QDRANT_URL")  # Qdrant Cloud URL
        qdrant_api_key = os.getenv("QDRANT_API_KEY")  # Qdrant Cloud API Key
        
        if not groq_api_key:
            logger.warning("⚠️ GROQ_API_KEY not set - LLM features will be limited")
        
        # Initialize Embedding Generator
        logger.info("📊 Initializing embedding generator...")
        app_state.embedding_generator = EmbeddingGenerator()
        app_state.embedding_generator.load_model()
        logger.info("✅ Embedding generator ready")
        
        # Initialize PDF Processor
        logger.info("📄 Initializing PDF processor...")
        app_state.pdf_processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
        logger.info("✅ PDF processor ready")
        
        # Initialize Vector Store (Qdrant Cloud)
        logger.info("🗄️ Connecting to Qdrant Cloud...")
        app_state.vector_store = VectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        app_state.vector_store.connect()
        
        # Create collections for different knowledge domains
        collections = [
            "phishing_insights",      # Campaign analytics
            "company_knowledge",      # Organization info
            "phishing_general",       # General phishing knowledge
            "pdf_documents"          # PDF documents
        ]
        
        for collection in collections:
            app_state.vector_store.create_collection(
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
            app_state.llm_orchestrator = LLMOrchestrator(
                api_key=groq_api_key,
                rag_retriever=app_state.rag_retriever
            )
            logger.info("✅ LLM orchestrator ready")
        
        app_state.initialized = True
        logger.info("✨ System fully initialized and ready!")
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        raise
    
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

# CORS Middleware - Allow all origins for now (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """Health check endpoint"""
    components = {
        "embedding_generator": app_state.embedding_generator is not None,
        "vector_store": app_state.vector_store is not None,
        "rag_retriever": app_state.rag_retriever is not None,
        "llm_orchestrator": app_state.llm_orchestrator is not None,
    }
    
    collections = []
    if app_state.vector_store:
        try:
            # Get all collections from Qdrant
            collections_info = app_state.vector_store.client.get_collections()
            collections = [col.name for col in collections_info.collections]
        except Exception as e:
            logger.error(f"Error fetching collections: {e}")
    
    return {
        "status": "healthy" if app_state.initialized else "initializing",
        "components": components,
        "collections": collections
    }

@app.post("/query_agent", response_model=QueryResponse, tags=["Query"])
async def query_agent(request: QueryRequest):
    """
    Main query endpoint - Handles natural language questions
    Supports queries about:
    - Phishing campaigns (click rates, risk analysis)
    - Company knowledge (who we are, what we do)
    - General phishing information (tactics, defenses)
    """
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="System not fully initialized")
    
    if not app_state.llm_orchestrator:
        raise HTTPException(status_code=503, detail="LLM not available - check GROQ_API_KEY")
    
    try:
        logger.info(f"📥 Query received: {request.query[:100]}...")
        
        # Process query through orchestrator
        response_text, sources = app_state.llm_orchestrator.process_query(
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
        processor.load_data()
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
        app_state.vector_store.add_documents(
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
        app_state.vector_store.add_documents(
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
        app_state.vector_store.add_documents(
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
        app_state.vector_store.add_documents(
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
        info = app_state.vector_store.get_collection_info(collection_name)
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
        results = app_state.vector_store.get_all_documents(
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
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Disable in production
        log_level="info"
    )