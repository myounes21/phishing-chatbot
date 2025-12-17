"""
Embeddings Module - Lightweight API-based approach
Uses Hugging Face Inference API - no local model downloads
"""

import numpy as np
from typing import List, Union, Dict, Any
import logging
import os
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using Hugging Face Inference API
    No local model downloads - pure API calls
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator
        
        Args:
            model_name: HuggingFace model for embeddings
        """
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        self.dimension = None
        
        # Model dimensions (common models)
        self.known_dimensions = {
            'BAAI/bge-small-en-v1.5': 384,
            'BAAI/bge-base-en-v1.5': 768,
            'sentence-transformers/all-MiniLM-L6-v2': 384,
            'sentence-transformers/all-mpnet-base-v2': 768,
        }
        
    def load_model(self):
        """Verify API connection and get embedding dimension"""
        try:
            logger.info(f"Connecting to HuggingFace API: {self.model_name}")
            
            if not self.api_key:
                logger.warning("⚠️ HUGGINGFACE_API_KEY not set - API may be rate limited")
            
            # Test API and get dimension
            test_embedding = self._call_api("test")
            if test_embedding is not None:
                self.dimension = len(test_embedding)
                logger.info(f"✅ API connected. Dimension: {self.dimension}")
            else:
                # Use known dimension or default
                self.dimension = self.known_dimensions.get(self.model_name, 384)
                logger.warning(f"⚠️ Using default dimension: {self.dimension}")
                
        except Exception as e:
            logger.error(f"❌ API connection error: {e}")
            self.dimension = self.known_dimensions.get(self.model_name, 384)
            logger.warning(f"⚠️ Using fallback dimension: {self.dimension}")
    
    def _call_api(self, text: Union[str, List[str]]) -> Union[np.ndarray, None]:
        """Call HuggingFace Inference API"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # Handle nested arrays (some models return [[...]])
                if isinstance(result, list):
                    if isinstance(result[0], list) and isinstance(result[0][0], list):
                        # Nested: take mean of token embeddings
                        return np.array([np.mean(r, axis=0) for r in result])
                    elif isinstance(result[0], list):
                        return np.array(result)
                    else:
                        return np.array(result)
                return np.array(result)
            else:
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None
    
    def encode_text(self, text: Union[str, List[str]], 
                    normalize: bool = True,
                    batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text via API
        
        Args:
            text: Single text string or list of texts
            normalize: Whether to normalize embeddings
            batch_size: Batch size for API calls
            
        Returns:
            Numpy array of embeddings
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._call_api(batch)
            
            if embeddings is None:
                raise Exception("Failed to get embeddings from API")
            
            all_embeddings.append(embeddings)
        
        # Combine batches
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)
        
        # Return single embedding if input was single text
        if is_single:
            return embeddings[0] if len(embeddings.shape) > 1 else embeddings
        
        return embeddings
    
    def encode_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embeddings to insight dictionaries"""
        texts = [insight['text'] for insight in insights]
        
        if not texts:
            return insights

        logger.info(f"Generating embeddings for {len(texts)} insights")
        embeddings = self.encode_text(texts)
        
        for insight, embedding in zip(insights, embeddings):
            insight['embedding'] = embedding.tolist()
        
        logger.info(f"✅ Generated embeddings for {len(insights)} insights")
        return insights
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        emb1 = self.encode_text(text1, normalize=True)
        emb2 = self.encode_text(text2, normalize=True)
        return float(np.dot(emb1, emb2))
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.dimension is None:
            self.load_model()
        return self.dimension or 384


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.load_model()
    
    text = "Finance department shows high vulnerability to phishing"
    embedding = generator.encode_text(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
