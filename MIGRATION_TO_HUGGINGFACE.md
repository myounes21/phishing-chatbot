# Migration from Mixedbread to Hugging Face Embeddings

## Summary

The embedding system has been migrated from Mixedbread AI to Hugging Face sentence-transformers using the `BAAI/bge-small-en-v1.5` model.

## Changes Made

### 1. Embedding Library
- **Removed:** `mixedbread-ai` package
- **Added:** `sentence-transformers` and `torch` packages

### 2. Environment Variables
- **Removed:** `MIXEDBREAD_API_KEY` (no longer required)
- **Added:** 
  - `HUGGINGFACE_API_KEY` (optional - only needed for private models)
  - `EMBEDDING_MODEL` (optional - defaults to `BAAI/bge-small-en-v1.5`)

### 3. Model Details
- **Model:** `BAAI/bge-small-en-v1.5`
- **Dimension:** 384 (changed from 1024)
- **API Key:** Not required for public models
- **Device:** Automatically uses GPU if available, falls back to CPU

## Updated .env File

Update your `.env` file:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional - Embedding Configuration
HUGGINGFACE_API_KEY=your_hf_token_here  # Optional - only for private models
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  # Optional - defaults to this

# Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# Server Configuration
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO
ENVIRONMENT=production
ALLOWED_ORIGINS=*
```

## Installation

Update your dependencies:

```bash
pip install -r requirements.txt --upgrade
```

This will:
- Remove `mixedbread-ai`
- Install `sentence-transformers` and `torch`

## Benefits

1. **No API Key Required** - Public Hugging Face models work without authentication
2. **Local Processing** - Embeddings are generated locally (faster, no API rate limits)
3. **GPU Support** - Automatically uses GPU if available for faster processing
4. **Smaller Embeddings** - 384 dimensions vs 1024 (more efficient storage)
5. **Open Source** - Full control over the embedding model

## Model Information

- **Model Name:** `BAAI/bge-small-en-v1.5`
- **Provider:** Beijing Academy of Artificial Intelligence (BAAI)
- **Type:** Sentence Transformer
- **Dimension:** 384
- **Language:** English
- **Performance:** High-quality embeddings optimized for retrieval tasks

## Migration Steps

1. **Update Dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Update .env File:**
   - Remove `MIXEDBREAD_API_KEY`
   - Add `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` (optional)
   - Add `HUGGINGFACE_API_KEY` only if using private models

3. **Restart Server:**
   ```bash
   uvicorn main:app --reload
   ```

4. **Recreate Collections (if needed):**
   - The vector dimension changed from 1024 to 384
   - Existing collections with 1024 dimensions need to be recreated
   - New data will automatically use 384 dimensions

## Important Notes

### Vector Dimension Change
- **Old:** 1024 dimensions (Mixedbread)
- **New:** 384 dimensions (BAAI/bge-small-en-v1.5)
- **Impact:** Existing collections with old dimension need to be recreated

### Recreating Collections

If you have existing data in Qdrant with 1024 dimensions, you'll need to:

1. Export your data (if needed)
2. Delete old collections
3. Restart the server (collections will be recreated with 384 dimensions)
4. Re-upload your data

Or manually update collection dimensions in Qdrant Cloud dashboard.

### API Key

- **HUGGINGFACE_API_KEY** is optional for public models
- Only required if you're using private/gated models
- Get one at: https://huggingface.co/settings/tokens

## Testing

Test the new embedding system:

```bash
python embeddings.py
```

Or test via the API:

```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Model Download Issues
If the model fails to download:
- Check internet connection
- Verify model name is correct: `BAAI/bge-small-en-v1.5`
- Try setting `HUGGINGFACE_API_KEY` if behind a firewall

### GPU Issues
If you want to force CPU usage:
```python
# In embeddings.py, change:
self.device = 'cpu'  # Force CPU
```

### Dimension Mismatch
If you see dimension errors:
- Ensure collections are recreated with 384 dimensions
- Check that `vector_size` matches the model dimension

## Performance

- **Speed:** Faster than API calls (no network latency)
- **Batch Processing:** Efficient batch encoding
- **GPU Acceleration:** Automatic GPU usage if available
- **Memory:** Lower memory footprint (384 vs 1024 dimensions)

## Support

For issues or questions:
1. Check server logs for detailed error messages
2. Verify model name in `.env` file
3. Test with `python embeddings.py`
4. Check Hugging Face model page: https://huggingface.co/BAAI/bge-small-en-v1.5
