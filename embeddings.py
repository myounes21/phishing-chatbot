"""
Embeddings Module - Lightweight API-based approach
Uses Hugging Face Inference API - no local model downloads
"""

import numpy as np
from typing import List, Union, Dict, Any
import time
import logging
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        
        # HuggingFace Router API (new endpoint as of 2024)
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        
        if not self.api_key:
            logger.error("❌ HUGGINGFACE_API_KEY required for HuggingFace Inference API")
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
            logger.info(f"API URL: {self.api_url}")
            
            if not self.api_key:
                logger.error("❌ HUGGINGFACE_API_KEY not set - API calls will fail!")
                self.dimension = self.known_dimensions.get(self.model_name, 384)
                return
            
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
    
    def _call_api(self, text: Union[str, List[str]], retries: int = 3) -> Union[np.ndarray, None]:
        """Call HuggingFace Inference API with retry logic"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        
        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    return self._parse_embedding_response(result, text)
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    logger.warning(f"Model loading, attempt {attempt + 1}/{retries}...")
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"API error {response.status_code}: {response.text[:300]}")
                    return None
                    
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return None
        
        return None
    
    def _parse_embedding_response(self, result: Any, original_input: Union[str, List[str]]) -> np.ndarray:
        """Parse the embedding response from HuggingFace API"""
        is_single = isinstance(original_input, str)
        
        try:
            # Result can be: [float...], [[float...]], or [[[float...]]]
            if not isinstance(result, list) or len(result) == 0:
                logger.error(f"Unexpected response format: {type(result)}")
                return None
            
            # Check depth of nesting
            if isinstance(result[0], (int, float)):
                # Single embedding: [float, float, ...]
                return np.array(result)
            
            elif isinstance(result[0], list):
                if len(result[0]) == 0:
                    logger.error("Empty embedding returned")
                    return None
                    
                if isinstance(result[0][0], (int, float)):
                    # Batch of embeddings: [[float...], [float...]]
                    return np.array(result)
                
                elif isinstance(result[0][0], list):
                    # Token-level embeddings: [[[float...]]] - take mean
                    embeddings = []
                    for item in result:
                        if isinstance(item[0], list):
                            # Mean pool token embeddings
                            embeddings.append(np.mean(item, axis=0))
                        else:
                            embeddings.append(item)
                    return np.array(embeddings)
            
            logger.error(f"Could not parse response: {str(result)[:200]}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing embedding response: {e}")
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
