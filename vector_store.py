"""
Vector Store Module - Qdrant Cloud Integration
Supports multiple collections for different knowledge domains
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
    Manages vector storage and retrieval using Qdrant Cloud
    Supports multiple collections for different knowledge domains
    """
    
    def __init__(self, 
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 vector_size: int = 384):
        """
        Initialize the vector store
        
        Args:
            url: Qdrant Cloud URL (e.g., https://xyz.qdrant.io)
            api_key: Qdrant Cloud API key
            vector_size: Dimension of vectors (default: 384 for MiniLM)
        """
        self.url = url
        self.api_key = api_key
        self.vector_size = vector_size
        self.client = None
        
    def connect(self):
        """Connect to Qdrant Cloud or fallback to in-memory"""
        
        # Try Qdrant Cloud first
        if self.url and self.api_key:
            try:
                logger.info(f"🔗 Connecting to Qdrant Cloud at {self.url}")
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=10
                )
                # Test connection
                self.client.get_collections()
                logger.info("✅ Successfully connected to Qdrant Cloud")
                return
                
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to Qdrant Cloud: {e}")
                logger.info("Falling back to in-memory storage...")
        
        # Fallback to in-memory storage
        try:
            self.client = QdrantClient(":memory:")
            logger.info("✅ Using in-memory Qdrant storage")
            logger.warning("⚠️ Data will not persist after restart")
            logger.info("💡 To use Qdrant Cloud, set QDRANT_URL and QDRANT_API_KEY")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant: {e}")
            raise RuntimeError("Could not initialize vector storage")
    
    def create_collection(self, 
                         collection_name: str,
                         recreate: bool = False):
        """
        Create a collection if it doesn't exist
        
        Args:
            collection_name: Name of the collection
            recreate: Whether to recreate if it exists
        """
        if self.client is None:
            self.connect()
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(
                c.name == collection_name for c in collections
            )
            
            if collection_exists and recreate:
                logger.info(f"🗑️ Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
                collection_exists = False
            
            if not collection_exists:
                logger.info(f"📦 Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Collection '{collection_name}' created")
            else:
                logger.info(f"ℹ️ Collection '{collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"❌ Error with collection {collection_name}: {e}")
            raise
    
    def add_documents(self, 
                     documents: List[Dict[str, Any]],
                     collection_name: str = "phishing_insights"):
        """
        Add documents with embeddings to a specific collection
        
        Args:
            documents: List of document dictionaries with 'embedding' field
            collection_name: Target collection name
        """
        if self.client is None:
            self.connect()
        
        try:
            points = []
            
            for doc in documents:
                # Generate unique ID if not present
                point_id = doc.get('id') or str(uuid.uuid4())
                
                # Prepare payload (exclude embedding)
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
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            
            logger.info(f"✅ Added {len(points)} documents to '{collection_name}'")
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to {collection_name}: {e}")
            raise
    
    def search(self, 
               query_vector: List[float],
               collection_name: str = "phishing_insights",
               top_k: int = 5,
               score_threshold: Optional[float] = 0.5,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in a collection
        
        Args:
            query_vector: Query embedding vector
            collection_name: Collection to search
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with scores and payloads
        """
        if self.client is None:
            self.connect()
        
        try:
            # Prepare filter
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
                collection_name=collection_name,
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
                    'payload': result.payload
                })
            
            logger.info(f"🔍 Found {len(formatted_results)} results in '{collection_name}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Search error in {collection_name}: {e}")
            return []
    
    def search_multi_collection(self,
                               query_vector: List[float],
                               collections: List[str],
                               top_k_per_collection: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple collections
        
        Args:
            query_vector: Query embedding
            collections: List of collection names to search
            top_k_per_collection: Results per collection
            
        Returns:
            Dictionary mapping collection name to results
        """
        results = {}
        
        for collection in collections:
            try:
                collection_results = self.search(
                    query_vector=query_vector,
                    collection_name=collection,
                    top_k=top_k_per_collection
                )
                results[collection] = collection_results
            except Exception as e:
                logger.warning(f"⚠️ Could not search collection '{collection}': {e}")
                results[collection] = []
        
        return results
    
    def get_all_documents(self,
                         collection_name: str = "phishing_insights",
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve documents from a collection
        
        Args:
            collection_name: Collection to retrieve from
            limit: Maximum number of documents
            
        Returns:
            List of documents
        """
        if self.client is None:
            self.connect()
        
        try:
            results = self.client.scroll(
                collection_name=collection_name,
                limit=limit
            )[0]
            
            documents = [point.payload for point in results]
            logger.info(f"📥 Retrieved {len(documents)} documents from '{collection_name}'")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents from {collection_name}: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        if self.client is None:
            self.connect()
        
        try:
            info = self.client.get_collection(collection_name)
            
            vectors_config = info.config.params.vectors
            vector_size = 0
            distance = "unknown"
            
            if hasattr(vectors_config, 'size'):
                vector_size = vectors_config.size
                distance = vectors_config.distance
            elif isinstance(vectors_config, dict):
                first_key = list(vectors_config.keys())[0]
                vector_size = vectors_config[first_key].size
                distance = vectors_config[first_key].distance
            
            return {
                'name': collection_name,
                'vector_size': vector_size,
                'distance': str(distance),
                'points_count': info.points_count
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting info for {collection_name}: {e}")
            return {}
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        if self.client is None:
            self.connect()
        
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"🗑️ Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"❌ Error deleting collection {collection_name}: {e}")
            raise


class RAGRetriever:
    """
    Enhanced RAG Retriever with multi-collection support
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
                        collection: Optional[str] = None,
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query text
            collection: Specific collection or None for auto-detect
            top_k: Number of contexts to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.encode_text(query)
        
        # If specific collection requested
        if collection:
            results = self.vector_store.search(
                query_vector=query_embedding.tolist(),
                collection_name=collection,
                top_k=top_k
            )
            return results
        
        # Auto-detect: Search across all collections
        collections = ["phishing_insights", "company_knowledge", "phishing_general", "pdf_documents"]
        all_results = self.vector_store.search_multi_collection(
            query_vector=query_embedding.tolist(),
            collections=collections,
            top_k_per_collection=3
        )
        
        # Combine and sort by relevance
        combined = []
        for coll_name, coll_results in all_results.items():
            for result in coll_results:
                result['collection'] = coll_name
                combined.append(result)
        
        # Sort by score and take top_k
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:top_k]
    
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
            collection = result.get('collection', 'unknown')
            
            text = payload.get('text', '')
            title = payload.get('title', payload.get('campaign_name', ''))
            
            context_parts.append(
                f"[Context {i}] (Relevance: {score:.2f}, Source: {collection})\n"
                f"Title: {title}\n"
                f"Content: {text}\n"
            )
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    # Test connection to Qdrant Cloud
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url or not api_key:
        print("⚠️ QDRANT_URL and QDRANT_API_KEY not set")
        print("Using in-memory storage for testing...")
    
    vector_store = VectorStore(url=url, api_key=api_key)
    vector_store.connect()
    
    # Test creating collections
    vector_store.create_collection("test_collection", recreate=True)
    
    print("✅ Vector store test completed")