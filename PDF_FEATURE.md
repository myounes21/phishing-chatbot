# 📄 PDF Document Support - Feature Guide

## ✅ What's New

The system now supports uploading PDF documents, extracting text, and making them searchable for answering questions!

---

## 🎯 Features

1. **PDF Upload** - Upload PDF files via API or web interface
2. **Text Extraction** - Automatically extracts text from PDF pages
3. **Smart Chunking** - Splits large documents into manageable chunks (500 chars with 50 char overlap)
4. **Vector Storage** - Stores PDF content in vector database for semantic search
5. **Query Support** - Answer questions about PDF content using natural language

---

## 📤 How to Upload PDFs

### Option 1: Web Interface (Easiest)

1. Open `upload.html` in your browser
2. Scroll to "📄 Upload PDF Document" section
3. Fill in:
   - **Document Title** (required)
   - **Category** (optional, e.g., "policy", "report", "manual")
   - **Description** (optional)
   - **PDF File** (required)
4. Click "Upload PDF Document"

### Option 2: API Endpoint

**Endpoint:** `POST /upload/pdf_document`

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/upload/pdf_document" \
  -F "file=@document.pdf" \
  -F "title=Security Policy 2024" \
  -F "category=policy" \
  -F "description=Company security policy document"
```

**Using Python:**
```python
import requests

url = "http://localhost:8000/upload/pdf_document"

with open('document.pdf', 'rb') as f:
    files = {'file': ('document.pdf', f, 'application/pdf')}
    data = {
        'title': 'Security Policy 2024',
        'category': 'policy',
        'description': 'Company security policy document'
    }
    
    response = requests.post(url, files=files, data=data)
    print(response.json())
```

**Using JavaScript:**
```javascript
const formData = new FormData();
formData.append('file', pdfFile);
formData.append('title', 'Security Policy 2024');
formData.append('category', 'policy');
formData.append('description', 'Company security policy');

const response = await fetch('http://localhost:8000/upload/pdf_document', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result);
```

---

## 🔍 How to Query PDF Content

Once uploaded, PDFs are automatically searchable through the query endpoint:

**Example Queries:**
- "What does the security policy say about password requirements?"
- "Summarize the incident response procedure from the manual"
- "What are the key points in the compliance report?"

**Using API:**
```bash
curl -X POST "http://localhost:8000/query_agent" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the security policy say about password requirements?",
    "include_sources": true
  }'
```

The system will:
1. Automatically detect PDF-related queries
2. Search the `pdf_documents` collection
3. Retrieve relevant chunks from your PDFs
4. Generate comprehensive answers using the LLM

---

## 🏗️ Technical Details

### PDF Processing Flow

```
PDF File Upload
    ↓
pdf_processor.py: extract_text_from_pdf()
    - Uses pdfplumber to extract text from each page
    ↓
pdf_processor.py: chunk_text()
    - Splits text into 500-character chunks
    - Adds 50-character overlap between chunks
    - Preserves context across chunks
    ↓
embeddings.py: encode_text()
    - Converts chunks to 384-dimensional vectors
    ↓
vector_store.py: add_documents()
    - Stores in "pdf_documents" collection in Qdrant
    ↓
Ready for semantic search!
```

### Chunking Strategy

- **Chunk Size:** 500 characters (configurable)
- **Overlap:** 50 characters between chunks
- **Method:** 
  1. First splits by paragraphs (double newlines)
  2. If chunks still too large, splits by sentences
  3. Preserves document structure and context

### Storage Structure

Each PDF chunk is stored with:
```python
{
    'id': 'document_title_chunk_0',
    'text': 'Extracted text content...',
    'title': 'Document Title',
    'category': 'policy',
    'chunk_index': 0,
    'source_type': 'pdf_document',
    'filename': 'document.pdf',
    'file_size': 12345,
    'description': 'Optional description',
    'embedding': [0.123, -0.456, ...]  # 384 numbers
}
```

---

## 📋 Supported PDF Types

✅ **Supported:**
- Text-based PDFs
- Multi-page documents
- PDFs with tables (text extraction)
- PDFs with basic formatting

❌ **Not Supported:**
- Image-only PDFs (scanned documents without OCR)
- Encrypted/Password-protected PDFs
- PDFs with complex layouts (may have extraction issues)

---

## 🔧 Configuration

### Adjust Chunk Size

In `main.py`, line 104:
```python
app_state.pdf_processor = PDFProcessor(
    chunk_size=500,      # Increase for longer chunks
    chunk_overlap=50     # Increase for more overlap
)
```

**Recommendations:**
- **Small chunks (300-500):** Better for precise answers, more chunks
- **Large chunks (800-1000):** Better for context, fewer chunks
- **Overlap (50-100):** Prevents losing context at chunk boundaries

---

## 🎯 Use Cases

### 1. Policy Documents
Upload company policies, procedures, and guidelines. Ask questions like:
- "What is our remote work policy?"
- "How do we handle data breaches?"
- "What are the password requirements?"

### 2. Technical Documentation
Upload manuals, guides, and specifications:
- "How do I configure the firewall?"
- "What are the API endpoints?"
- "Explain the authentication flow"

### 3. Reports & Analysis
Upload reports and ask for summaries:
- "Summarize the Q4 security report"
- "What were the main findings?"
- "What recommendations were made?"

### 4. Compliance Documents
Upload compliance and regulatory documents:
- "What are our GDPR requirements?"
- "What data retention policies apply?"
- "What are the audit requirements?"

---

## 🐛 Troubleshooting

### Issue: "No text could be extracted from PDF"

**Causes:**
- PDF is image-based (scanned document)
- PDF is encrypted/password-protected
- PDF is corrupted

**Solutions:**
- Use OCR software to convert scanned PDFs to text
- Remove password protection
- Try a different PDF file

### Issue: "PDF file is empty"

**Solution:** Make sure you're uploading a valid PDF file, not empty or corrupted.

### Issue: Chunks are too small/large

**Solution:** Adjust `chunk_size` and `chunk_overlap` in `main.py` (line 104).

### Issue: Poor search results

**Solutions:**
- Upload PDFs with clear, structured text
- Use descriptive titles and categories
- Ensure PDF text is selectable (not just images)

---

## 📊 API Response Example

**Upload Response:**
```json
{
    "status": "success",
    "message": "PDF document 'Security Policy 2024' processed successfully",
    "collection": "pdf_documents",
    "chunks_added": 45
}
```

**Query Response:**
```json
{
    "query": "What does the security policy say about passwords?",
    "response": "# Password Requirements\n\nAccording to the Security Policy 2024 document...",
    "sources": [
        {
            "content": "Passwords must be at least 12 characters long...",
            "relevance": 0.92,
            "collection": "pdf_documents",
            "title": "Security Policy 2024"
        }
    ],
    "metadata": {
        "query_length": 45,
        "response_length": 523,
        "sources_count": 3,
        "collection_used": "pdf_documents"
    }
}
```

---

## 🚀 Next Steps

1. **Upload your first PDF:**
   ```bash
   curl -X POST "http://localhost:8000/upload/pdf_document" \
     -F "file=@your_document.pdf" \
     -F "title=My Document" \
     -F "category=general"
   ```

2. **Query it:**
   ```bash
   curl -X POST "http://localhost:8000/query_agent" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is in my document?"}'
   ```

3. **Check collection:**
   ```bash
   curl http://localhost:8000/collections/pdf_documents/info
   ```

---

## 📝 Files Modified/Created

- ✅ `requirements.txt` - Added `pdfplumber>=0.10.0`
- ✅ `pdf_processor.py` - New PDF processing module
- ✅ `main.py` - Added PDF upload endpoint and initialization
- ✅ `llm_orchestrator.py` - Updated query detection for PDFs
- ✅ `vector_store.py` - Added pdf_documents to auto-search
- ✅ `upload.html` - Added PDF upload form

---

**🎉 PDF support is now fully integrated! Upload your documents and start asking questions!**






