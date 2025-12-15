"""
Embeddings Module
Handles text embedding generation using sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using sentence-transformers models
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self.dimension = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self.dimension = len(test_embedding)
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]], 
                    normalize: bool = True,
                    batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single text string or list of texts
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Convert single string to list
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=normalize
            )
            
            # Return single embedding if input was single text
            if is_single:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def encode_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to insight dictionaries
        
        Args:
            insights: List of insight dictionaries
            
        Returns:
            List of insights with added 'embedding' field
        """
        if self.model is None:
            self.load_model()
        
        # Extract texts
        texts = [insight['text'] for insight in insights]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} insights")
        embeddings = self.encode_text(texts)
        
        # Add embeddings to insights
        for insight, embedding in zip(insights, embeddings):
            insight['embedding'] = embedding.tolist()
        
        logger.info(f"Successfully generated embeddings for {len(insights)} insights")
        return insights
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if self.model is None:
            self.load_model()
        
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
    
    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model
        
        Returns:
            Embedding dimension
        """
        if self.dimension is None:
            if self.model is None:
                self.load_model()
        
        return self.dimension


if __name__ == "__main__":
    # Test the embedding generator
    generator = EmbeddingGenerator()
    generator.load_model()
    
    # Test single text
    text = "Finance department shows high vulnerability to phishing attacks"
    embedding = generator.encode_text(text)
    print(f"Single text embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test multiple texts
    texts = [
        "Finance department shows high vulnerability",
        "Sales team has low click rate",
        "IT department reports most phishing emails"
    ]
    embeddings = generator.encode_text(texts)
    print(f"\nMultiple texts embedding shape: {embeddings.shape}")
    
    # Test similarity
    similarity = generator.compute_similarity(texts[0], texts[1])
    print(f"\nSimilarity between text 1 and 2: {similarity:.4f}")
    
    similarity = generator.compute_similarity(texts[0], texts[0])
    print(f"Similarity between text 1 and itself: {similarity:.4f}")
