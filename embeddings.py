"""
Embeddings Module
Handles text embedding generation using Hugging Face sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Dict, Any
import logging
import os
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using Hugging Face sentence-transformers models
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the Hugging Face model to use (defaults to env var or BAAI/bge-small-en-v1.5)
        """
        # Get model name from environment or use default
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
        self.model = None
        self.dimension = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """Initialize the Hugging Face sentence-transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Load the model
            # Note: Hugging Face API key is optional for public models
            # If you have a private model, you can use: token=os.getenv('HUGGINGFACE_API_KEY')
            hf_token = os.getenv('HUGGINGFACE_API_KEY')
            
            if hf_token:
                logger.info("Using Hugging Face API token")
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    use_auth_token=hf_token
                )
            else:
                logger.info("Loading public model (no API token required)")
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
            
            # Get embedding dimension by encoding a test string
            try:
                test_embedding = self.model.encode("test", normalize_embeddings=True)
                self.dimension = len(test_embedding)
                logger.info(f"✅ Model loaded successfully. Embedding dimension: {self.dimension}")
            except Exception as e:
                logger.error(f"❌ Could not determine embedding dimension: {e}")
                # Default dimension for BAAI/bge-small-en-v1.5 is 384
                self.dimension = 384
                logger.warning(f"⚠️ Using default dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]], 
                    normalize: bool = True,
                    batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single text string or list of texts
            normalize: Whether to normalize embeddings (default: True)
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
            
            # Encode texts using sentence-transformers
            # The model handles batching internally, but we can specify batch_size
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            # Convert to numpy array if not already
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Return single embedding if input was single text
            if is_single:
                return embeddings[0] if len(embeddings.shape) > 1 else embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error encoding text: {e}")
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
        
        if not texts:
            return insights

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} insights")
        embeddings = self.encode_text(texts)
        
        # Add embeddings to insights
        for insight, embedding in zip(insights, embeddings):
            insight['embedding'] = embedding.tolist()
        
        logger.info(f"✅ Successfully generated embeddings for {len(insights)} insights")
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
        
        emb1 = self.encode_text(text1, normalize=True)
        emb2 = self.encode_text(text2, normalize=True)
        
        # Cosine similarity (already normalized, so just dot product)
        similarity = np.dot(emb1, emb2)
        
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

            # If still None, try to encode a test string
            if self.dimension is None:
                try:
                    test_embedding = self.model.encode("test", normalize_embeddings=True)
                    self.dimension = len(test_embedding)
                except Exception as e:
                    logger.error(f"❌ Failed to determine embedding dimension: {e}")
                    # Default dimension for BAAI/bge-small-en-v1.5 is 384
                    self.dimension = 384
                    logger.warning(f"⚠️ Using default dimension: {self.dimension}")
        
        return self.dimension


if __name__ == "__main__":
    # Test the embedding generator
    # Note: HUGGINGFACE_API_KEY is optional for public models
    generator = EmbeddingGenerator()
    
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
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
