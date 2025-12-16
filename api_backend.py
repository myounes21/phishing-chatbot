import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil
from typing import List

# Import custom modules
from data_processor import PhishingDataProcessor
from document_processor import DocumentProcessor
from insight_generator import InsightGenerator
from embeddings import EmbeddingGenerator
from vector_store import VectorStore, RAGRetriever
from llm_orchestrator import LLMOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    response: str

# Global instances
orchestrator = None
vector_store = None
embedding_generator = None
data_processor = None
document_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    global orchestrator, vector_store, embedding_generator, data_processor, document_processor
    try:
        logger.info("Initializing system components...")

        # Configuration
        csv_path = os.getenv("PHISHING_DATA_PATH", "sample_phishing_data.csv")
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            logger.warning("GROQ_API_KEY not found in environment variables.")

        # Initialize Embedding Generator
        logger.info("Initializing embedding generator...")
        embedding_generator = EmbeddingGenerator()
        embedding_generator.load_model()

        # Initialize Vector Store
        logger.info("Setting up vector database...")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        vector_store = VectorStore(host=qdrant_host, port=qdrant_port)
        vector_store.connect()
        # Ensure collection exists, but don't force recreate to preserve knowledge base
        vector_store.create_collection(recreate=False)

        # Initialize Data Processor (Phishing)
        # We only process if file exists, but always initialize the processor
        document_processor = DocumentProcessor()

        if os.path.exists(csv_path):
             logger.info(f"Loading data from {csv_path}...")
             data_processor = PhishingDataProcessor(csv_path)
             try:
                 data_processor.load_data()

                 # Generate insights and index them if needed
                 # Ideally we should check if they are already indexed, but for now we re-index on startup for the phishing part
                 logger.info("Generating analytical insights...")
                 insight_generator = InsightGenerator(data_processor)
                 insights = insight_generator.generate_all_insights()

                 logger.info("Creating text embeddings for insights...")
                 insights_with_embeddings = embedding_generator.encode_insights(insights)

                 logger.info("Indexing insights...")
                 vector_store.add_documents(insights_with_embeddings)

             except Exception as e:
                 logger.error(f"Error processing phishing data: {e}")
        else:
            logger.warning(f"Phishing data file not found at {csv_path}")
            data_processor = PhishingDataProcessor(csv_path) # Initialize empty

        # Initialize RAG Retriever
        logger.info("Configuring RAG system...")
        rag_retriever = RAGRetriever(
            vector_store,
            embedding_generator
        )

        # Initialize LLM Orchestrator
        logger.info("Connecting to Groq LLM...")
        orchestrator = LLMOrchestrator(
            api_key=groq_api_key,
            data_processor=data_processor,
            rag_retriever=rag_retriever
        )

        logger.info("System initialized successfully!")

    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        # Log error but allow app to start
        pass

    yield

    # Cleanup logic (if any) would go here
    logger.info("Shutting down system...")


app = FastAPI(title="Phishing Campaign Agent API", lifespan=lifespan)


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        response = orchestrator.process_query(request.user_query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("general_knowledge") # org_knowledge, general_knowledge, phishing_data
):
    """
    Upload a file to the knowledge base or update phishing data
    """
    if not vector_store or not embedding_generator:
         raise HTTPException(status_code=503, detail="System components not initialized")

    filename = file.filename
    temp_path = f"temp_{filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Received file: {filename}, type: {doc_type}")

        if doc_type == "phishing_data" and filename.endswith(".csv"):
            # Update phishing data processor
            global data_processor, orchestrator

            # Re-initialize processor with new file
            data_processor = PhishingDataProcessor(temp_path)
            data_processor.load_data()

            # Generate insights
            insight_generator = InsightGenerator(data_processor)
            insights = insight_generator.generate_all_insights()

            # Encode and Store
            insights_with_embeddings = embedding_generator.encode_insights(insights)
            vector_store.add_documents(insights_with_embeddings)

            # Update orchestrator
            orchestrator.data_processor = data_processor

            return {"status": "success", "message": "Phishing data updated and insights generated", "count": len(insights)}

        else:
            # Handle generic documents
            chunks = document_processor.process_file(temp_path, doc_type)

            # Encode
            texts = [chunk['text'] for chunk in chunks]
            embeddings = embedding_generator.encode_text(texts)

            for chunk, emb in zip(chunks, embeddings):
                chunk['embedding'] = emb.tolist()

            # Store
            vector_store.add_documents(chunks)

            return {"status": "success", "message": f"Document {filename} processed and added to knowledge base", "chunks": len(chunks)}

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health_check():
    return {"status": "ok", "initialized": orchestrator is not None}

@app.get("/collections")
async def get_collections():
    if not vector_store:
         raise HTTPException(status_code=503, detail="Vector store not initialized")

    return vector_store.get_collection_info()

@app.get("/stats/{collection}")
async def get_collection_stats(collection: str):
    # Currently we only support one collection, so we return info for that
    if not vector_store:
         raise HTTPException(status_code=503, detail="Vector store not initialized")

    if collection != vector_store.collection_name:
        raise HTTPException(status_code=404, detail="Collection not found")

    return vector_store.get_collection_info()
