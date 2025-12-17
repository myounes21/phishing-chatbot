# Railway Deployment Guide

This guide will help you deploy the Phishing Campaign Analyzer chatbot API to Railway.

## Prerequisites

1. **Railway Account** - Sign up at https://railway.app
2. **Groq API Key** - Get free at https://console.groq.com
3. **Mixedbread API Key** - Get at https://www.mixedbread.ai/
4. **Qdrant Cloud Account** - Sign up at https://cloud.qdrant.io (free tier available)

## Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure your code is pushed to GitHub (or GitLab/Bitbucket).

### 2. Create New Railway Project

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo" (or your Git provider)
4. Select your repository

### 3. Configure Environment Variables

In Railway dashboard, go to your service → Variables tab, and add:

```env
# Required - LLM API
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Required - Embeddings API
MIXEDBREAD_API_KEY=your_mixedbread_api_key_here

# Required - Vector Database
QDRANT_URL=https://your-cluster-id.region.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# Server Configuration
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO
ENVIRONMENT=production

# CORS - Allow all origins for chatbot integration
ALLOWED_ORIGINS=*

# Optional - RAG Configuration
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.5
LLM_TEMPERATURE=0.4
LLM_MAX_TOKENS=4096

# Optional - Collection Names
COLLECTION_PHISHING=phishing_insights
COLLECTION_COMPANY=company_knowledge
COLLECTION_GENERAL=phishing_general
```

### 4. Configure Build Settings

Railway should auto-detect Python, but verify:

1. Go to Settings → Build
2. **Build Command:** (leave empty - Railway auto-detects)
3. **Start Command:** `python -m uvicorn main:app --host 0.0.0.0 --port $PORT`

### 5. Deploy

Railway will automatically:
1. Install dependencies from `requirements.txt`
2. Start the server
3. Expose your API at a public URL

### 6. Verify Deployment

1. **Check Health:**
   ```
   https://your-app.up.railway.app/health
   ```

2. **Test Query:**
   ```bash
   curl -X POST https://your-app.up.railway.app/query_agent \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello, are you working?"}'
   ```

3. **View Logs:**
   - Go to Railway dashboard → Deployments → View Logs
   - Look for initialization messages

## Integration with Website

### Frontend Integration Example

```javascript
// Simple fetch example
async function askChatbot(question) {
    const response = await fetch('https://your-app.up.railway.app/query_agent', {
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
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    return {
        answer: data.response,
        sources: data.sources
    };
}

// Usage
askChatbot("What is phishing?")
    .then(result => {
        console.log("Answer:", result.answer);
        console.log("Sources:", result.sources);
    })
    .catch(error => {
        console.error("Error:", error);
    });
```

### React Example

```jsx
import { useState } from 'react';

function Chatbot() {
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    
    const API_URL = 'https://your-app.up.railway.app';
    
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        
        try {
            const res = await fetch(`${API_URL}/query_agent`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, include_sources: true })
            });
            
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
            
            const data = await res.json();
            setResponse(data.response);
        } catch (error) {
            setResponse(`Error: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <form onSubmit={handleSubmit}>
            <input 
                value={query} 
                onChange={e => setQuery(e.target.value)}
                placeholder="Ask a question..."
            />
            <button type="submit" disabled={loading}>
                {loading ? 'Sending...' : 'Send'}
            </button>
            {response && <div>{response}</div>}
        </form>
    );
}
```

## Troubleshooting

### 503 Service Unavailable

1. **Check Railway Logs:**
   - Go to Railway dashboard → Deployments → View Logs
   - Look for initialization errors

2. **Check Health Endpoint:**
   ```
   https://your-app.up.railway.app/health
   ```
   This shows which components failed to initialize

3. **Verify Environment Variables:**
   - Make sure all API keys are set correctly
   - Check for typos in variable names
   - Ensure no extra spaces in values

### Common Issues

**"System not fully initialized"**
- Check that all required API keys are set
- Verify API keys are valid (not expired)
- Check Railway logs for specific error messages

**"LLM not available"**
- Verify `GROQ_API_KEY` is set
- Check Groq API key is valid at https://console.groq.com
- Ensure key has proper permissions

**"Failed to initialize embedding generator"**
- Verify `MIXEDBREAD_API_KEY` is set
- Check Mixedbread API key is valid
- Ensure you have API credits/quota

**"Could not connect to Qdrant"**
- Verify `QDRANT_URL` and `QDRANT_API_KEY` are set
- Check Qdrant Cloud cluster is running
- Verify network connectivity from Railway

## Monitoring

### Railway Metrics

Railway provides:
- CPU/Memory usage
- Request logs
- Deployment history
- Error tracking

### Health Monitoring

Set up periodic health checks:
```bash
# Check health every 5 minutes
curl https://your-app.up.railway.app/health
```

## Scaling

Railway automatically scales based on traffic. For high-traffic chatbots:

1. **Upgrade Railway Plan** - More resources
2. **Add Caching** - Reduce API calls
3. **Rate Limiting** - Already configured via `MAX_REQUESTS_PER_MINUTE`
4. **CDN** - Use Cloudflare for static assets

## Security Best Practices

1. **API Keys:**
   - Never commit `.env` file to Git
   - Use Railway's environment variables (encrypted)
   - Rotate keys regularly

2. **CORS:**
   - Set `ALLOWED_ORIGINS` to specific domains in production
   - Don't use `*` in production if possible

3. **Rate Limiting:**
   - Configure `MAX_REQUESTS_PER_MINUTE` appropriately
   - Monitor for abuse

## Support

If you encounter issues:

1. Check Railway logs
2. Visit `/health` endpoint
3. Review `TROUBLESHOOTING.md`
4. Check API provider status pages:
   - Groq: https://status.groq.com
   - Mixedbread: Check their status page
   - Qdrant: https://status.qdrant.io

---

**Your API URL:** `https://your-app.up.railway.app`

**Useful Endpoints:**
- `/` - API info
- `/health` - System status
- `/docs` - API documentation (Swagger UI)
- `/query_agent` - Main chatbot endpoint
