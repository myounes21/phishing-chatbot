import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import custom modules
from data_processor import PhishingDataProcessor
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

# Global orchestrator instance
orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    global orchestrator
    try:
        logger.info("Initializing system components...")

        # Configuration
        csv_path = os.getenv("PHISHING_DATA_PATH", "sample_phishing_data.csv")
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            logger.warning("GROQ_API_KEY not found in environment variables.")

        if not os.path.exists(csv_path):
             logger.error(f"Data file not found at {csv_path}")
             # We try to use the sample data if the env var path failed
             if csv_path != "sample_phishing_data.csv" and os.path.exists("sample_phishing_data.csv"):
                 logger.info("Falling back to sample_phishing_data.csv")
                 csv_path = "sample_phishing_data.csv"
             elif os.path.exists("temp_upload.csv"):
                 # Fallback to temp_upload.csv if it exists
                 csv_path = "temp_upload.csv"
                 logger.info(f"Using {csv_path} instead.")

        # Step 1: Load and process data
        logger.info(f"Loading data from {csv_path}...")
        data_processor = PhishingDataProcessor(csv_path)
        data_processor.load_data()

        # Step 2: Generate insights
        logger.info("Generating analytical insights...")
        insight_generator = InsightGenerator(data_processor)
        insights = insight_generator.generate_all_insights()

        # Step 3: Create embeddings
        logger.info("Creating text embeddings...")
        embedding_generator = EmbeddingGenerator()
        insights_with_embeddings = embedding_generator.encode_insights(insights)

        # Step 4: Setup vector store
        logger.info("Setting up vector database...")
        # Get Qdrant config from env or default
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

        vector_store = VectorStore(host=qdrant_host, port=qdrant_port)
        vector_store.connect()
        vector_store.create_collection(recreate=True)
        vector_store.add_insights(insights_with_embeddings)

        # Step 5: Create RAG retriever
        logger.info("Configuring RAG system...")
        rag_retriever = RAGRetriever(
            vector_store,
            embedding_generator
        )

        # Step 6: Initialize LLM orchestrator
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


@app.post("/query_agent")
async def query_agent(request: QueryRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        response = orchestrator.process_query(request.user_query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "initialized": orchestrator is not None}
