"""
Embeddings Module
Handles text embedding generation using mixedbread-ai
"""

from mixedbread_ai.client import MixedbreadAI
import numpy as np
from typing import List, Union, Dict, Any
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using mixedbread-ai models
    """
    
    def __init__(self, model_name: str = "mixedbread-ai/mxbai-embed-large-v1"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the mixedbread-ai model to use
        """
        self.model_name = model_name
        self.client = None
        self.dimension = None
        
    def load_model(self):
        """Initialize the mixedbread-ai client"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            api_key = os.getenv('MIXEDBREAD_API_KEY')
            if not api_key:
                logger.warning("MIXEDBREAD_API_KEY environment variable is not set")

            self.client = MixedbreadAI(api_key=api_key)
            
            # Get embedding dimension by embedding a test string
            try:
                # We need to determine dimension.
                # If the API call fails (e.g. no key), we might not be able to set dimension.
                # But typically we need dimension for vector store initialization.
                response = self.client.embeddings(
                    model=self.model_name,
                    input="test"
                )
                if response.data and len(response.data) > 0:
                    self.dimension = len(response.data[0].embedding)

                logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
            except Exception as e:
                logger.warning(f"Could not determine embedding dimension on load: {e}")
                # We don't raise here to allow instantiation even if API is not reachable yet,
                # but subsequent calls will fail if client is not working.
            
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
        if self.client is None:
            self.load_model()
        
        try:
            # Convert single string to list
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            all_embeddings = []

            # Process in batches
            # The SDK handles lists, but we might want to respect batch_size if provided
            # and if the list is very large.
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                response = self.client.embeddings(
                    model=self.model_name,
                    input=batch_texts,
                    normalized=normalize
                )

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            # Convert to numpy array
            embeddings = np.array(all_embeddings)
            
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
        if self.client is None:
            self.load_model()
        
        # Extract texts
        texts = [insight['text'] for insight in insights]
        
        if not texts:
            return insights

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
        if self.client is None:
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
            if self.client is None:
                self.load_model()

            # If still None, try to fetch again
            if self.dimension is None:
                try:
                    response = self.client.embeddings(
                        model=self.model_name,
                        input="test"
                    )
                    if response.data and len(response.data) > 0:
                        self.dimension = len(response.data[0].embedding)
                except Exception as e:
                    logger.error(f"Failed to determine embedding dimension: {e}")
                    raise
        
        return self.dimension


if __name__ == "__main__":
    # Test the embedding generator
    # Note: Requires MIXEDBREAD_API_KEY environment variable
    generator = EmbeddingGenerator()
    
    # We catch the error to print a friendly message if API key is missing
    try:
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

    except Exception as e:
        print(f"Test failed (likely due to missing API key): {e}")
