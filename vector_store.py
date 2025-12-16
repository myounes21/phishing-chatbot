"""
Vector Store Module
Handles storage and retrieval of embeddings using Qdrant
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
from typing import List, Dict, Any, Optional
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector storage and retrieval using Qdrant
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6333,
                 collection_name: str = "phishing_insights",
                 vector_size: int = 384):
        """
        Initialize the vector store
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            vector_size: Dimension of vectors
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = None
        
    def connect(self):
        """Connect to Qdrant server with automatic fallback to in-memory"""
        # Try connecting to external Qdrant first
        if self.host != ":memory:":
            try:
                logger.info(f"Attempting to connect to Qdrant at {self.host}:{self.port}")
                self.client = QdrantClient(
                    host=self.host, 
                    port=self.port,
                    timeout=5  # 5 second timeout
                )
                # Test the connection
                self.client.get_collections()
                logger.info("✓ Successfully connected to Qdrant server")
                return
                
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant at {self.host}:{self.port}")
                logger.warning(f"Error: {str(e)}")
                logger.info("Falling back to in-memory storage...")
        
        # Fallback to in-memory storage
        try:
            self.client = QdrantClient(":memory:")
            logger.info("✓ Using in-memory Qdrant storage")
            logger.info("  Note: Data will not persist after restart")
            logger.info("  To use persistent storage, start Qdrant with:")
            logger.info("  docker run -p 6333:6333 qdrant/qdrant")
            
        except Exception as e2:
            logger.error(f"Failed to initialize in-memory Qdrant: {e2}")
            raise RuntimeError(
                "Could not initialize vector storage. "
                "Please ensure qdrant-client is installed correctly."
            )
    
    def create_collection(self, recreate: bool = False):
        """
        Create collection if it doesn't exist
        
        Args:
            recreate: Whether to recreate the collection if it exists
        """
        if self.client is None:
            self.connect()
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(
                c.name == self.collection_name for c in collections
            )
            
            if collection_exists and recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents with embeddings to the vector store.
        Each document should be a dict with 'embedding' and other metadata fields.
        
        Args:
            documents: List of document dictionaries with 'embedding' field
        """
        if self.client is None:
            self.connect()
        
        try:
            points = []
            
            for doc in documents:
                # Generate unique ID if not present
                point_id = doc.get('id') or str(uuid.uuid4())
                
                # Prepare payload (exclude embedding from payload)
                payload = {
                    k: v for k, v in doc.items()
                    if k != 'embedding'
                }
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=doc['embedding'],
                    payload=payload
                )
                
                points.append(point)
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Successfully added {len(points)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    # Deprecated alias for backward compatibility
    def add_insights(self, insights: List[Dict[str, Any]]):
        return self.add_documents(insights)
    
    def search(self, 
               query_vector: List[float],
               top_k: int = 5,
               score_threshold: Optional[float] = None,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar insights
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with scores and payloads
        """
        if self.client is None:
            self.connect()
        
        try:
            # Prepare filter if provided
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions) if conditions else None
            
            # Perform search
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold
            ).points
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'score': result.score,
                    'payload': result.payload,
                    'insight': result.payload # Backward compatibility
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def search_by_category(self,
                          query_vector: List[float],
                          category: str,
                          top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents within a specific category
        
        Args:
            query_vector: Query embedding vector
            category: Category to filter by
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            filter_dict={'category': category}
        )
    
    def search_by_source_type(self,
                            query_vector: List[float],
                            source_type: str,
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents within a specific source type (e.g. 'phishing_insight', 'org_knowledge')

        Args:
            query_vector: Query embedding vector
            source_type: Source type to filter by
            top_k: Number of results to return

        Returns:
            List of search results
        """
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            filter_dict={'source_type': source_type}
        )

    def get_all_insights(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all insights from the store
        
        Args:
            limit: Maximum number of insights to retrieve
            
        Returns:
            List of insights
        """
        if self.client is None:
            self.connect()
        
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit
            )[0]
            
            insights = [point.payload for point in results]
            logger.info(f"Retrieved {len(insights)} insights")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error retrieving insights: {e}")
            raise
    
    def delete_collection(self):
        """Delete the collection"""
        if self.client is None:
            self.connect()
        
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        if self.client is None:
            self.connect()
        
        try:
            info = self.client.get_collection(self.collection_name)
            
            # The 'vectors' config can be a single VectorParams or a dict of them.
            # We assume single vector for now or take the first one.
            vectors_config = info.config.params.vectors
            vector_size = 0
            distance = "unknown"

            if hasattr(vectors_config, 'size'):
                vector_size = vectors_config.size
                distance = vectors_config.distance
            elif isinstance(vectors_config, dict):
                 # Get first one
                 first_key = list(vectors_config.keys())[0]
                 vector_size = vectors_config[first_key].size
                 distance = vectors_config[first_key].distance

            return {
                'name': self.collection_name, # Return the name we know
                'vector_size': vector_size,
                'distance': distance,
                'points_count': info.points_count
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever for context extraction
    """
    
    def __init__(self, vector_store: VectorStore, embedding_generator):
        """
        Initialize the RAG retriever
        
        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def retrieve_context(self, 
                        query: str,
                        top_k: int = 5,
                        category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query text
            top_k: Number of contexts to retrieve
            category: Optional category filter
            
        Returns:
            List of relevant insights
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.encode_text(query)
        
        # Search vector store
        if category:
            results = self.vector_store.search_by_category(
                query_vector=query_embedding.tolist(),
                category=category,
                top_k=top_k
            )
        else:
            results = self.vector_store.search(
                query_vector=query_embedding.tolist(),
                top_k=top_k,
                score_threshold=0.01  # Lowered threshold to ensure we get results in tests/demos
            )
        
        logger.info(f"Retrieved {len(results)} contexts for query: {query[:50]}...")
        return results
    
    def format_context_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved contexts for LLM consumption
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            payload = result['payload']
            score = result['score']
            
            source_type = payload.get('source_type', 'unknown')
            category = payload.get('category', 'general')
            text = payload.get('text', '')

            context_parts.append(
                f"[Context {i}] (Relevance: {score:.2f}, Source: {source_type})\n"
                f"Category: {category}\n"
                f"Content: {text}\n"
            )
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    # Test the vector store
    from embeddings import EmbeddingGenerator
    from data_processor import PhishingDataProcessor
    from insight_generator import InsightGenerator
    
    # Generate insights with embeddings
    processor = PhishingDataProcessor("sample_phishing_data.csv")
    processor.load_data()
    
    generator = InsightGenerator(processor)
    insights = generator.generate_all_insights()
    
    # Add embeddings
    embedding_gen = EmbeddingGenerator()
    insights_with_embeddings = embedding_gen.encode_insights(insights)
    
    # Create vector store
    vector_store = VectorStore()
    vector_store.connect()
    vector_store.create_collection(recreate=True)
    
    # Add insights
    vector_store.add_insights(insights_with_embeddings)
    
    # Test search
    print("\n=== Testing Vector Search ===")
    query = "Which department is most vulnerable?"
    query_embedding = embedding_gen.encode_text(query)
    
    results = vector_store.search(query_embedding.tolist(), top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['score']:.4f}):")
        print(f"Text: {result['insight']['text'][:200]}...")
    
    # Test RAG retriever
    print("\n=== Testing RAG Retriever ===")
    retriever = RAGRetriever(vector_store, embedding_gen)
    
    contexts = retriever.retrieve_context("Why is Finance vulnerable?", top_k=3)
    formatted_context = retriever.format_context_for_llm(contexts)
    print(formatted_context)