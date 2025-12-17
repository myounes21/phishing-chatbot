# Quick Start Guide

## ✅ What I've Fixed

1. **Improved Error Handling** - Better error messages that tell you exactly what's wrong
2. **Enhanced Health Endpoint** - Now shows which components failed and environment variable status
3. **Better CORS Configuration** - Optimized for web chatbot integration
4. **Resilient Initialization** - Better handling of partial failures
5. **Diagnostic Tools** - Added `/health` and `/test` endpoints for troubleshooting

## 🚀 Quick Start

### 1. Check Environment Variables

```bash
python check_env.py
```

This will verify all your API keys are set correctly.

### 2. Start the Server

```bash
uvicorn main:app --reload
```

### 3. Check Health Status

Visit: `http://localhost:8000/health`

This will show you:
- Which components are initialized
- Which environment variables are set
- What's missing (if anything)

### 4. Test the API

```bash
# Simple test
curl http://localhost:8000/test

# Health check
curl http://localhost:8000/health

# Query test
curl -X POST http://localhost:8000/query_agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, are you working?"}'
```

## 🔍 Troubleshooting 503 Error

If you're still getting a 503 error:

### Step 1: Check Server Logs

Look at the terminal where you ran `uvicorn main:app`. You should see initialization messages. Look for:
- ❌ Error messages
- ⚠️ Warning messages
- ✅ Success messages

### Step 2: Check Health Endpoint

Visit `http://localhost:8000/health` and check:
- `status`: Should be "healthy"
- `components`: All should be `true`
- `environment`: All should be "set"

### Step 3: Common Issues

**Issue: "GROQ_API_KEY" shows as "set" but LLM not initialized**
- Your API key might be invalid
- Check: `GROQ_API_KEY=mey_key` - this looks like a placeholder/typo
- Get a real key from https://console.groq.com

**Issue: "MIXEDBREAD_API_KEY" shows as "set" but embedding generator failed**
- Your API key might be invalid
- Check the key is correct
- Verify at https://www.mixedbread.ai/

**Issue: Qdrant connection failed**
- Check your `QDRANT_URL` format
- Should be: `https://cluster-id.region.cloud.qdrant.io`
- Verify your `QDRANT_API_KEY` is valid

## 🌐 Web Integration

### For Website Chatbot

The API is ready for web integration. Example:

```javascript
const API_URL = 'https://your-app.up.railway.app';

async function askChatbot(question) {
    const response = await fetch(`${API_URL}/query_agent`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: question,
            include_sources: true
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP ${response.status}`);
    }
    
    return await response.json();
}

// Usage
askChatbot("What is phishing?")
    .then(data => {
        console.log("Answer:", data.response);
        console.log("Sources:", data.sources);
    })
    .catch(error => {
        console.error("Error:", error.message);
    });
```

### CORS Configuration

The API is configured to allow all origins (`ALLOWED_ORIGINS=*`). For production, you can restrict this:

```env
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

## 📋 API Endpoints

- `GET /` - API information
- `GET /health` - Health check and diagnostics
- `GET /test` - Simple connectivity test
- `POST /query_agent` - Main chatbot query endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)

## 🚂 Railway Deployment

See `RAILWAY_DEPLOYMENT.md` for detailed Railway deployment instructions.

Key points:
1. Push code to GitHub
2. Connect to Railway
3. Set environment variables in Railway dashboard
4. Deploy!

## 📝 Next Steps

1. **Verify API Keys** - Make sure they're real keys, not placeholders
2. **Check Logs** - Look for initialization errors
3. **Test Health** - Visit `/health` to see what's working
4. **Test Query** - Try a simple query to verify everything works
5. **Deploy** - Follow Railway deployment guide

## 🆘 Still Having Issues?

1. Check `TROUBLESHOOTING.md` for detailed help
2. Check Railway logs (if deployed)
3. Verify all API keys are valid
4. Check `/health` endpoint for component status
5. Review server logs for specific error messages

---

**Remember:** The 503 error means initialization failed. Check `/health` endpoint to see exactly what failed!
