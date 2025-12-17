# Troubleshooting Guide

## 503 Service Temporarily Unavailable Error

If you're getting a **503 Service Temporarily Unavailable** error, it means the system is not fully initialized. Here's how to fix it:

### Step 1: Check the Health Endpoint

Visit: `http://localhost:8000/health` (or your server URL)

This will show you which components are initialized:
```json
{
  "status": "healthy" | "error - check logs",
  "components": {
    "embedding_generator": true/false,
    "vector_store": true/false,
    "rag_retriever": true/false,
    "llm_orchestrator": true/false
  },
  "collections": [...],
  "message": "System is ready" | "System initialization failed - check logs"
}
```

### Step 2: Check Server Logs

Look at the terminal where you ran `uvicorn main:app`. You should see initialization messages:

✅ **Good logs:**
```
🚀 Initializing Phishing Campaign Analyzer & Knowledge Chatbot...
📊 Initializing embedding generator...
✅ Embedding generator ready
📄 Initializing PDF processor...
✅ PDF processor ready
🗄️ Connecting to Qdrant Cloud...
✅ Collection 'phishing_insights' ready
🔍 Setting up RAG retriever...
✅ RAG retriever ready
🤖 Connecting to Groq LLM...
✅ LLM orchestrator ready
✨ System fully initialized and ready!
```

❌ **Bad logs (look for these errors):**
```
❌ Initialization error: ...
❌ Failed to initialize embedding generator: ...
❌ Failed to initialize LLM orchestrator: ...
```

### Step 3: Common Issues and Fixes

#### Issue 1: Missing GROQ_API_KEY

**Error:** `LLM not available - check GROQ_API_KEY`

**Fix:**
1. Get a free API key from https://console.groq.com
2. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Restart the server

#### Issue 2: Missing MIXEDBREAD_API_KEY

**Error:** `Failed to initialize embedding generator`

**Fix:**
1. Get an API key from https://www.mixedbread.ai/
2. Add to `.env` file:
   ```
   MIXEDBREAD_API_KEY=your_key_here
   ```
3. Restart the server

#### Issue 3: Qdrant Connection Failed

**Error:** `Could not connect to Qdrant Cloud`

**Fix:**
1. If using Qdrant Cloud, add to `.env`:
   ```
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your_key_here
   ```
2. If you don't have Qdrant Cloud, the system will fall back to in-memory storage (data won't persist)
3. Restart the server

#### Issue 4: Initialization Exception

**Error:** `System not fully initialized`

**Fix:**
1. Check all the above API keys are set
2. Check server logs for the specific error
3. Make sure all dependencies are installed: `pip install -r requirements.txt`
4. Try restarting the server

### Step 4: Verify Environment Variables

Create a `.env` file in the project root with:

```env
# Required for query functionality
GROQ_API_KEY=your_groq_api_key_here

# Required for embeddings
MIXEDBREAD_API_KEY=your_mixedbread_api_key_here

# Optional - Qdrant Cloud (falls back to in-memory if not set)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# Optional - Server configuration
PORT=8000
HOST=0.0.0.0
ENVIRONMENT=development
LOG_LEVEL=info
```

### Step 5: Test the System

1. **Check health:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test a query:**
   ```bash
   curl -X POST http://localhost:8000/query_agent \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello, are you working?"}'
   ```

### Still Having Issues?

1. **Check Python version:** Should be 3.10+
   ```bash
   python --version
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Check for port conflicts:**
   ```bash
   # Windows
   netstat -ano | findstr :8000
   
   # Linux/Mac
   lsof -i :8000
   ```

4. **Run with verbose logging:**
   ```bash
   LOG_LEVEL=debug uvicorn main:app --reload
   ```

5. **Check the FastAPI docs:**
   Visit: `http://localhost:8000/docs` to see all available endpoints

---

## Other Common Errors

### "Cannot connect to server"
- Make sure the server is running: `uvicorn main:app --reload`
- Check the API URL in your frontend matches the server URL
- Check firewall/antivirus isn't blocking the connection

### "Request timed out"
- The server might be slow to respond
- Check if Groq API is experiencing issues
- Try reducing `RAG_TOP_K` in environment variables

### "No text could be extracted from PDF"
- The PDF might be image-based or encrypted
- Try a different PDF file
- Check PDF file isn't corrupted

---

## Getting Help

1. Check server logs for detailed error messages
2. Visit `/health` endpoint to see component status
3. Test individual endpoints using `/docs` (Swagger UI)
4. Verify all environment variables are set correctly
