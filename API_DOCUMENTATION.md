# API Documentation - Phishing Campaign Analyzer & Knowledge Chatbot

**Base URL:** `https://phishing-chatbot-production.up.railway.app`  
**API Version:** 2.0.0  
**Interactive Docs:** `/docs` (Swagger UI)

---

## Table of Contents

1. [System Endpoints](#system-endpoints)
2. [Query Endpoints](#query-endpoints)
3. [Upload Endpoints](#upload-endpoints)
4. [Session Management](#session-management)
5. [Feedback Endpoints](#feedback-endpoints)
6. [Collection & Campaign Endpoints](#collection--campaign-endpoints)
7. [Error Handling](#error-handling)
8. [Integration Examples](#integration-examples)

---

## System Endpoints

### GET `/`
**Description:** Root endpoint - API information

**Response:**
```json
{
  "service": "Phishing Campaign Analyzer & Knowledge Chatbot",
  "version": "2.0.0",
  "status": "running",
  "docs": "/docs"
}
```

**cURL Example:**
```bash
curl https://phishing-chatbot-production.up.railway.app/
```

---

### GET `/health`
**Description:** Health check - Check system status and component initialization

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "embedding_generator": true,
    "vector_store": true,
    "rag_retriever": true,
    "llm_orchestrator": true
  },
  "collections": [
    "pdf_documents",
    "phishing_general",
    "phishing_insights",
    "company_knowledge"
  ],
  "message": "System is ready",
  "environment": {
    "GROQ_API_KEY": "set",
    "HUGGINGFACE_API_KEY": "set",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "QDRANT_URL": "set",
    "QDRANT_API_KEY": "set"
  }
}
```

**Use Case:** Check if API is ready before making queries

**cURL Example:**
```bash
curl https://phishing-chatbot-production.up.railway.app/health
```

---

### GET `/test`
**Description:** Simple connectivity test

**Response:**
```json
{
  "status": "ok",
  "message": "API is responding",
  "initialized": true,
  "timestamp": "2024-12-17T16:30:00"
}
```

**cURL Example:**
```bash
curl https://phishing-chatbot-production.up.railway.app/test
```

---

## Query Endpoints

### POST `/query_agent`
**Description:** Main query endpoint - Ask questions and get AI responses with RAG (Retrieval-Augmented Generation)

**Request Body:**
```json
{
  "query": "What is the click rate for Finance department?",
  "collection": null,
  "campaign_id": null,
  "include_sources": true
}
```

**Parameters:**
- `query` (required, string): Your natural language question
- `collection` (optional, string): `"phishing_insights"`, `"company_knowledge"`, `"phishing_general"`, or `null` for auto-detect
- `campaign_id` (optional, string): Filter by specific campaign ID
- `include_sources` (optional, boolean): Include source citations (default: `true`)

**Response:**
```json
{
  "query": "What is the click rate for Finance department?",
  "response": "Looking at your campaign data, I can see that the Finance department has a click rate of 25%, which makes it moderately vulnerable to phishing attacks. This is higher than the company average...",
  "sources": [
    {
      "content": "The Finance department shows a 25% click rate, making it moderately vulnerable...",
      "relevance": 0.92,
      "collection": "phishing_insights",
      "title": "Finance Department Analysis"
    }
  ],
  "metadata": {
    "query_length": 45,
    "response_length": 320,
    "sources_count": 1,
    "collection_used": "auto-detected"
  }
}
```

**cURL Example:**
```bash
curl -X POST https://phishing-chatbot-production.up.railway.app/query_agent \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is phishing?",
    "include_sources": true
  }'
```

**JavaScript Example:**
```javascript
const response = await fetch('https://phishing-chatbot-production.up.railway.app/query_agent', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'What is the click rate for Finance department?',
    include_sources: true
  })
});

const data = await response.json();
console.log(data.response);
console.log('Sources:', data.sources);
```

---

### POST `/query_stream`
**Description:** Streaming query endpoint - Real-time response via Server-Sent Events (SSE). Better UX as users see response being generated token-by-token.

**Request Body:**
```json
{
  "query": "Explain phishing attacks",
  "session_id": "optional-session-id",
  "collection": null,
  "include_sources": true
}
```

**Parameters:**
- `query` (required, string): Your question
- `session_id` (optional, string): Session ID for conversation memory
- `collection` (optional, string): Specific collection to query
- `include_sources` (optional, boolean): Include sources (default: `true`)

**Response Format (SSE Stream):**
```
data: {"type":"session","session_id":"abc123","message_id":"msg456"}

data: {"type":"typing","status":"start"}

data: {"type":"content","content":"Phishing"}

data: {"type":"content","content":" is"}

data: {"type":"content","content":" a"}

data: {"type":"content","content":" type"}

data: {"type":"done","sources":[...],"suggested_questions":[],"message_id":"msg456"}
```

**JavaScript Example:**
```javascript
const response = await fetch('https://phishing-chatbot-production.up.railway.app/query_stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    query: 'What is phishing?',
    include_sources: true 
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let fullResponse = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const text = decoder.decode(value);
  const lines = text.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      
      if (data.type === 'content') {
        fullResponse += data.content;
        // Display chunk to user
        displayChunk(data.content);
      } else if (data.type === 'done') {
        // Display sources and finish
        displaySources(data.sources);
      }
    }
  }
}
```

---

### GET `/quick_actions`
**Description:** Get pre-defined quick action buttons for common queries

**Response:**
```json
{
  "quick_actions": [
    {
      "id": "department_summary",
      "label": "📊 Department Summary",
      "query": "Give me a summary of click rates by department",
      "icon": "📊"
    },
    {
      "id": "risky_users",
      "label": "⚠️ Risky Users",
      "query": "Who are the top 5 riskiest users and why?",
      "icon": "⚠️"
    },
    {
      "id": "phishing_tactics",
      "label": "🧠 Phishing Tactics",
      "query": "What are the most common phishing tactics?",
      "icon": "🧠"
    },
    {
      "id": "defense_tips",
      "label": "🛡️ Defense Tips",
      "query": "How can we protect against phishing attacks?",
      "icon": "🛡️"
    },
    {
      "id": "campaign_overview",
      "label": "📈 Campaign Overview",
      "query": "Give me an overview of our phishing campaigns",
      "icon": "📈"
    },
    {
      "id": "about_company",
      "label": "🏢 About Us",
      "query": "Tell me about our company",
      "icon": "🏢"
    }
  ]
}
```

**cURL Example:**
```bash
curl https://phishing-chatbot-production.up.railway.app/quick_actions
```

---

## Upload Endpoints

### POST `/upload/phishing_campaign`
**Description:** Upload CSV file with phishing campaign simulation data. Automatically generates insights and stores in vector database.

**Request:** `multipart/form-data`
- `file` (file, required): CSV file
- `campaign_name` (string, required): Campaign name
- `campaign_description` (string, optional): Campaign description

**CSV Format Required:**
```csv
User_ID,Department,Template,Action,Response_Time_Sec
user1,Finance,Urgent Invoice,Clicked,45
user2,Sales,Password Reset,Ignored,120
user3,IT,Security Alert,Reported,30
```

**Required Columns:**
- `User_ID` - Unique user identifier
- `Department` - User's department
- `Template` - Phishing email template name
- `Action` - `Clicked`, `Ignored`, or `Reported`
- `Response_Time_Sec` - Time in seconds before action

**Response:**
```json
{
  "status": "success",
  "message": "Campaign 'Q4 2024' processed successfully",
  "collection": "phishing_insights",
  "chunks_added": 15,
  "campaign_id": "campaign_q4_2024"
}
```

**cURL Example:**
```bash
curl -X POST https://phishing-chatbot-production.up.railway.app/upload/phishing_campaign \
  -F "file=@campaign.csv" \
  -F "campaign_name=Q4 2024 Security Test" \
  -F "campaign_description=Quarterly awareness training"
```

**JavaScript Example:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('campaign_name', 'Q4 2024 Test');
formData.append('campaign_description', 'Quarterly training');

const response = await fetch('https://phishing-chatbot-production.up.railway.app/upload/phishing_campaign', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(`Uploaded ${data.chunks_added} insights`);
```

**What Happens:**
1. CSV is validated and loaded
2. Statistics calculated (click rates, department analysis, etc.)
3. Insights generated (department vulnerabilities, risky users, template effectiveness)
4. Each insight converted to text and embedded (384-dimensional vector)
5. Stored in `phishing_insights` collection
6. Now queryable via chatbot

---

### POST `/upload/company_knowledge`
**Description:** Upload company information (mission, values, services, team info)

**Request:** `multipart/form-data`
- `title` (string, required): Document title
- `content` (string, required): Text content (paragraphs separated by double newlines)
- `category` (string, optional): Category like "about", "mission", "services" (default: "general")

**Response:**
```json
{
  "status": "success",
  "message": "Company knowledge 'About Us' added successfully",
  "collection": "company_knowledge",
  "chunks_added": 3
}
```

**cURL Example:**
```bash
curl -X POST https://phishing-chatbot-production.up.railway.app/upload/company_knowledge \
  -F "title=About Our Company" \
  -F "content=We are a cybersecurity company founded in 2020. Our mission is to protect organizations from cyber threats..." \
  -F "category=about"
```

**Content Format:**
- Separate paragraphs with double newlines (`\n\n`)
- Each paragraph becomes a searchable chunk
- Example:
```
We are a cybersecurity company.

Our mission is to protect organizations.

We offer security training and consulting services.
```

---

### POST `/upload/phishing_general`
**Description:** Upload general phishing knowledge/education content

**Request:** `multipart/form-data`
- `title` (string, required): Document title
- `content` (string, required): Text content
- `topic` (string, optional): Topic like "tactics", "defense", "education" (default: "general")

**Response:**
```json
{
  "status": "success",
  "message": "Phishing knowledge 'Common Tactics' added successfully",
  "collection": "phishing_general",
  "chunks_added": 5
}
```

**cURL Example:**
```bash
curl -X POST https://phishing-chatbot-production.up.railway.app/upload/phishing_general \
  -F "title=Common Phishing Tactics" \
  -F "content=Phishing attacks commonly use urgency and fear tactics. Attackers impersonate trusted entities..." \
  -F "topic=tactics"
```

---

### POST `/upload/pdf_document`
**Description:** Upload PDF document (policies, reports, manuals, guides). Text is extracted, chunked, and made searchable.

**Request:** `multipart/form-data`
- `file` (file, required): PDF file
- `title` (string, required): Document title
- `category` (string, optional): Category like "policy", "report", "manual"
- `description` (string, optional): Brief description

**Response:**
```json
{
  "status": "success",
  "message": "PDF document 'Security Policy 2024' processed successfully",
  "collection": "pdf_documents",
  "chunks_added": 12
}
```

**cURL Example:**
```bash
curl -X POST https://phishing-chatbot-production.up.railway.app/upload/pdf_document \
  -F "file=@policy.pdf" \
  -F "title=Security Policy 2024" \
  -F "category=policy" \
  -F "description=Company security policy document"
```

**What Happens:**
1. PDF text extracted
2. Split into chunks (500 chars with 50 char overlap)
3. Each chunk embedded (384-dimensional vector)
4. Stored in `pdf_documents` collection
5. Now queryable via chatbot

---

## Session Management

### GET `/session/{session_id}`
**Description:** Get session information and conversation history

**Path Parameters:**
- `session_id` (string, required): Session ID

**Response:**
```json
{
  "session_id": "abc123",
  "created_at": "2024-12-17T10:00:00",
  "conversation_history": [
    {
      "message_id": "msg1",
      "query": "What is phishing?",
      "response": "Phishing is a type of cyber attack...",
      "timestamp": "2024-12-17T10:01:00"
    },
    {
      "message_id": "msg2",
      "query": "How can I protect myself?",
      "response": "Here are some ways to protect yourself...",
      "timestamp": "2024-12-17T10:02:00"
    }
  ],
  "message_count": 2
}
```

**cURL Example:**
```bash
curl https://phishing-chatbot-production.up.railway.app/session/abc123
```

**Use Case:** Retrieve conversation history for a user session

---

### DELETE `/session/{session_id}`
**Description:** Clear session conversation history

**Path Parameters:**
- `session_id` (string, required): Session ID

**Response:**
```json
{
  "status": "success",
  "message": "Session cleared"
}
```

**cURL Example:**
```bash
curl -X DELETE https://phishing-chatbot-production.up.railway.app/session/abc123
```

**Use Case:** Clear chat history when user starts new conversation

---

## Feedback Endpoints

### POST `/feedback`
**Description:** Submit feedback (thumbs up/down) for a bot response

**Request Body:**
```json
{
  "message_id": "msg123",
  "session_id": "session456",
  "rating": "up",
  "query": "What is phishing?",
  "response": "Phishing is a type of cyber attack..."
}
```

**Parameters:**
- `message_id` (required, string): Message ID from query response
- `session_id` (required, string): Session ID
- `rating` (required, string): `"up"` or `"down"`
- `query` (optional, string): Original query
- `response` (optional, string): Bot response

**Response:**
```json
{
  "status": "success",
  "message": "Thank you for your feedback!"
}
```

**cURL Example:**
```bash
curl -X POST https://phishing-chatbot-production.up.railway.app/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg123",
    "session_id": "session456",
    "rating": "up"
  }'
```

**Use Case:** Collect user feedback to improve response quality

---

### GET `/feedback/stats`
**Description:** Get feedback statistics

**Response:**
```json
{
  "total_feedback": 150,
  "up_votes": 120,
  "down_votes": 30,
  "satisfaction_rate": 80.0
}
```

**cURL Example:**
```bash
curl https://phishing-chatbot-production.up.railway.app/feedback/stats
```

**Use Case:** Monitor chatbot performance and user satisfaction

---

## Collection & Campaign Endpoints

### GET `/collections/{collection_name}/info`
**Description:** Get information about a specific collection

**Path Parameters:**
- `collection_name` (string, required): Collection name

**Available Collections:**
- `phishing_insights` - Campaign analytics and insights
- `company_knowledge` - Organization information
- `phishing_general` - General phishing knowledge
- `pdf_documents` - PDF documents

**Response:**
```json
{
  "name": "phishing_insights",
  "vector_size": 384,
  "distance": "Cosine",
  "points_count": 150
}
```

**cURL Example:**
```bash
curl https://phishing-chatbot-production.up.railway.app/collections/phishing_insights/info
```

---

### GET `/campaigns/list`
**Description:** List all uploaded campaigns

**Response:**
```json
{
  "total_campaigns": 3,
  "campaigns": [
    {
      "campaign_id": "campaign_q4_2024",
      "campaign_name": "Q4 2024 Security Test",
      "campaign_description": "Quarterly awareness training"
    },
    {
      "campaign_id": "campaign_q3_2024",
      "campaign_name": "Q3 2024 Security Test",
      "campaign_description": ""
    }
  ]
}
```

**cURL Example:**
```bash
curl https://phishing-chatbot-production.up.railway.app/campaigns/list
```

**Use Case:** Display list of campaigns in UI or filter queries by campaign

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| `200` | Success | Request completed successfully |
| `400` | Bad Request | Invalid request format, missing required fields |
| `404` | Not Found | Resource doesn't exist (session, collection) |
| `500` | Internal Server Error | Server-side processing error |
| `503` | Service Unavailable | System not initialized or component missing |

### Error Response Format
```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

**503 - LLM not available:**
```json
{
  "detail": "LLM not available. Please set GROQ_API_KEY environment variable. Get a free API key at https://console.groq.com"
}
```
**Solution:** Check Railway environment variables

**503 - System not initialized:**
```json
{
  "detail": "System not initialized"
}
```
**Solution:** Check `/health` endpoint for component status

**400 - Invalid file type:**
```json
{
  "detail": "Only CSV files are supported"
}
```
**Solution:** Ensure file extension matches endpoint requirements

**400 - Missing required columns:**
```json
{
  "detail": "Missing required columns: {'User_ID', 'Department'}"
}
```
**Solution:** Check CSV format matches required schema

---

## Integration Examples

### JavaScript/Fetch - Query Endpoint
```javascript
async function askQuestion(query) {
  const response = await fetch('https://phishing-chatbot-production.up.railway.app/query_agent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: query,
      include_sources: true
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  const data = await response.json();
  return {
    answer: data.response,
    sources: data.sources,
    metadata: data.metadata
  };
}

// Usage
const result = await askQuestion('What is phishing?');
console.log(result.answer);
```

### JavaScript/Fetch - Streaming Query
```javascript
async function streamQuestion(query, onChunk, onComplete) {
  const response = await fetch('https://phishing-chatbot-production.up.railway.app/query_stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, include_sources: true })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let sessionId = null;
  let messageId = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));

        if (data.type === 'session') {
          sessionId = data.session_id;
          messageId = data.message_id;
        } else if (data.type === 'content') {
          onChunk(data.content);
        } else if (data.type === 'done') {
          onComplete(data.sources, messageId);
        }
      }
    }
  }
}

// Usage
streamQuestion(
  'What is phishing?',
  (chunk) => console.log(chunk), // Display each chunk
  (sources, msgId) => console.log('Done!', sources)
);
```

### Python - Query Endpoint
```python
import requests

def ask_question(query, include_sources=True):
    url = 'https://phishing-chatbot-production.up.railway.app/query_agent'
    response = requests.post(
        url,
        json={
            'query': query,
            'include_sources': include_sources
        }
    )
    response.raise_for_status()
    return response.json()

# Usage
result = ask_question('What is phishing?')
print(result['response'])
print(f"Sources: {len(result['sources'])}")
```

### Python - Upload CSV
```python
import requests

def upload_campaign(csv_path, campaign_name, description=None):
    url = 'https://phishing-chatbot-production.up.railway.app/upload/phishing_campaign'
    
    with open(csv_path, 'rb') as f:
        files = {'file': f}
        data = {
            'campaign_name': campaign_name,
            'campaign_description': description or ''
        }
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()

# Usage
result = upload_campaign(
    'campaign.csv',
    'Q4 2024 Test',
    'Quarterly training'
)
print(f"Uploaded {result['chunks_added']} insights")
```

### Python - Upload Company Knowledge
```python
import requests

def upload_company_knowledge(title, content, category='general'):
    url = 'https://phishing-chatbot-production.up.railway.app/upload/company_knowledge'
    
    data = {
        'title': title,
        'content': content,
        'category': category
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()

# Usage
result = upload_company_knowledge(
    'About Our Company',
    'We are a cybersecurity company.\n\nOur mission is to protect organizations.',
    'about'
)
print(f"Added {result['chunks_added']} chunks")
```

---

## Quick Reference Table

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/` | GET | API info | No |
| `/health` | GET | System status | No |
| `/test` | GET | Connectivity test | No |
| `/query_agent` | POST | Ask questions (JSON) | No |
| `/query_stream` | POST | Ask questions (streaming) | No |
| `/quick_actions` | GET | Get quick actions | No |
| `/upload/phishing_campaign` | POST | Upload CSV campaign | No |
| `/upload/company_knowledge` | POST | Upload company info | No |
| `/upload/phishing_general` | POST | Upload phishing knowledge | No |
| `/upload/pdf_document` | POST | Upload PDF files | No |
| `/session/{id}` | GET | Get session history | No |
| `/session/{id}` | DELETE | Clear session | No |
| `/feedback` | POST | Submit feedback | No |
| `/feedback/stats` | GET | Feedback statistics | No |
| `/collections/{name}/info` | GET | Collection info | No |
| `/campaigns/list` | GET | List campaigns | No |

---

## Frontend Integration

### Using the Provided HTML Files

**Chat Interface:** `example.html`
- Pre-configured with Railway API URL
- Supports streaming responses
- Includes feedback buttons
- Quick action buttons
- Session management

**Upload Interface:** `upload.html`
- Upload CSV campaigns
- Upload company knowledge
- Upload PDF documents
- Upload phishing knowledge
- Pre-configured with Railway API URL

**To Use:**
1. Open `example.html` or `upload.html` in browser
2. API URL is pre-filled: `https://phishing-chatbot-production.up.railway.app`
3. Start using immediately!

---

## Best Practices

1. **Always check `/health` first** - Ensure system is ready before queries
2. **Use streaming endpoint** (`/query_stream`) for better UX
3. **Include session_id** - Maintains conversation context
4. **Handle errors gracefully** - Check response status codes
5. **Use appropriate collections** - Specify collection for faster, more accurate results
6. **Validate CSV format** - Ensure required columns before upload
7. **Chunk large content** - Split long documents into paragraphs for better search

---

## Support & Resources

- **Interactive API Docs:** `https://phishing-chatbot-production.up.railway.app/docs`
- **Health Check:** `https://phishing-chatbot-production.up.railway.app/health`
- **GitHub Repository:** (Your repo URL)
- **Railway Dashboard:** (Your Railway project URL)

---

**Last Updated:** December 2024  
**API Version:** 2.0.0
