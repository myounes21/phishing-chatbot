# 🔍 Code Explanation: How File Upload Works

This document explains **exactly which code handles file uploads** and how it works step-by-step.

---

## 📍 Main Upload Endpoints (in `main.py`)

### 1. Phishing Campaign CSV Upload

**Location:** `main.py` lines 257-326

```python
@app.post("/upload/phishing_campaign", response_model=UploadResponse, tags=["Upload"])
async def upload_phishing_campaign(
    file: UploadFile = File(...),           # ← Receives the CSV file
    campaign_name: str = Form(...),         # ← Receives campaign name
    campaign_description: Optional[str] = Form(None)  # ← Optional description
):
```

**Step-by-Step Flow:**

```python
# STEP 1: Read the uploaded file (line 277-278)
content = await file.read()              # Read file bytes
df = pd.read_csv(io.BytesIO(content))    # Convert to pandas DataFrame

# STEP 2: Save temporarily (line 281-282)
temp_path = f"temp_{campaign_name}.csv"
df.to_csv(temp_path, index=False)

# STEP 3: Process the data (line 285-287)
processor = PhishingDataProcessor(temp_path)
processor.load_data()                    # Loads CSV, validates columns

# STEP 4: Generate insights (line 290-291)
insight_generator = InsightGenerator(processor)
insights = insight_generator.generate_all_insights()  # Creates text insights

# STEP 5: Add campaign metadata (line 294-299)
campaign_id = f"campaign_{campaign_name.lower().replace(' ', '_')}"
for insight in insights:
    insight['campaign_id'] = campaign_id
    insight['campaign_name'] = campaign_name
    if campaign_description:
        insight['campaign_description'] = campaign_description

# STEP 6: Create embeddings (line 302)
insights_with_embeddings = app_state.embedding_generator.encode_insights(insights)

# STEP 7: Store in vector database (line 305-308)
app_state.vector_store.add_documents(
    documents=insights_with_embeddings,
    collection_name="phishing_insights"
)

# STEP 8: Cleanup temp file (line 311-312)
if os.path.exists(temp_path):
    os.remove(temp_path)
```

---

### 2. Company Knowledge Upload

**Location:** `main.py` lines 328-382

```python
@app.post("/upload/company_knowledge", response_model=UploadResponse, tags=["Upload"])
async def upload_company_knowledge(
    content: str = Form(...),    # ← Receives text content
    title: str = Form(...),      # ← Receives title
    category: str = Form("general")  # ← Receives category
):
```

**Step-by-Step Flow:**

```python
# STEP 1: Split content into chunks (line 345)
paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

# STEP 2: Create document objects (line 347-356)
documents = []
for i, para in enumerate(paragraphs):
    doc = {
        'id': f"{title.lower().replace(' ', '_')}_{i}",
        'text': para,
        'title': title,
        'category': category,
        'source_type': 'company_knowledge'
    }
    documents.append(doc)

# STEP 3: Create embeddings (line 359-363)
texts = [doc['text'] for doc in documents]
embeddings = app_state.embedding_generator.encode_text(texts)

for doc, emb in zip(documents, embeddings):
    doc['embedding'] = emb.tolist()

# STEP 4: Store in vector database (line 366-369)
app_state.vector_store.add_documents(
    documents=documents,
    collection_name="company_knowledge"
)
```

---

### 3. Phishing General Knowledge Upload

**Location:** `main.py` lines 384-438

**Same flow as company knowledge**, but stores in `"phishing_general"` collection.

---

## 🔧 Supporting Modules

### A. Data Processing (`data_processor.py`)

**What it does:** Analyzes the CSV data and calculates statistics

**Key Methods:**
- `load_data()` (line 32-49) - Loads CSV and validates columns
- `calculate_click_rates()` (line 51-72) - Calculates click rates by department
- `calculate_template_effectiveness()` (line 74-103) - Analyzes which templates work best
- `identify_high_risk_users()` (line 105-143) - Finds most vulnerable users
- `get_department_summary()` (line 177-229) - Gets stats for a department

**Example:**
```python
processor = PhishingDataProcessor("campaign.csv")
processor.load_data()                    # Loads CSV
click_rates = processor.calculate_click_rates()  # Returns: {'Finance': 0.325, 'IT': 0.15, ...}
```

---

### B. Insight Generation (`insight_generator.py`)

**What it does:** Converts raw statistics into human-readable text insights

**Key Methods:**
- `generate_department_insights()` (line 29-99) - Creates insights about departments
- `generate_template_insights()` - Creates insights about templates
- `generate_user_risk_insights()` - Creates insights about risky users
- `generate_all_insights()` - Generates all types of insights

**Example Output:**
```python
insight = {
    'text': "The Finance department shows a 32.5% click rate, making it critically vulnerable...",
    'category': 'department_vulnerability',
    'department': 'Finance',
    'severity': 'HIGH',
    'click_rate': 0.325,
    ...
}
```

---

### C. Embedding Generation (`embeddings.py`)

**What it does:** Converts text into numerical vectors (embeddings) for semantic search

**Key Methods:**
- `encode_text()` (line 47-85) - Converts text to embeddings
- `encode_insights()` (line 87-112) - Adds embeddings to insight dictionaries

**Example:**
```python
# Input: "Finance department shows high vulnerability"
# Output: [0.123, -0.456, 0.789, ...] (384 numbers)
embedding = embedding_generator.encode_text("Finance department shows high vulnerability")
```

**How it works:**
1. Uses `sentence-transformers/all-MiniLM-L6-v2` model
2. Converts text to 384-dimensional vector
3. These vectors allow semantic search (finding similar meanings, not just keywords)

---

### D. Vector Store (`vector_store.py`)

**What it does:** Stores embeddings in Qdrant Cloud (vector database)

**Key Methods:**
- `connect()` (line 42-72) - Connects to Qdrant Cloud
- `create_collection()` (line 74-114) - Creates a collection if needed
- `add_documents()` (line 116-170) - Stores documents with embeddings

**Example:**
```python
# Each document becomes a "point" in Qdrant
point = PointStruct(
    id="unique_id",
    vector=[0.123, -0.456, ...],  # The embedding
    payload={                      # The original data
        'text': "Finance department shows...",
        'department': 'Finance',
        'click_rate': 0.325,
        ...
    }
)
vector_store.add_documents([point], collection_name="phishing_insights")
```

---

## 🔄 Complete Upload Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User uploads CSV file via POST /upload/phishing_campaign │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. main.py: upload_phishing_campaign()                       │
│    - Receives file as UploadFile                             │
│    - Reads bytes: await file.read()                          │
│    - Converts to DataFrame: pd.read_csv()                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. data_processor.py: PhishingDataProcessor                 │
│    - Validates CSV columns                                   │
│    - Calculates statistics (click rates, templates, etc.)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. insight_generator.py: InsightGenerator                     │
│    - Converts stats to human-readable text                   │
│    - Creates insight objects with metadata                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. embeddings.py: EmbeddingGenerator                         │
│    - Takes insight text                                      │
│    - Converts to 384-dimensional vector                      │
│    - Adds 'embedding' field to each insight                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. vector_store.py: VectorStore                              │
│    - Creates PointStruct objects (id + vector + payload)     │
│    - Uploads to Qdrant Cloud                                 │
│    - Stores in "phishing_insights" collection                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Return success response                                   │
│    - Status: "success"                                       │
│    - Chunks added: 45                                       │
│    - Campaign ID: "campaign_q4_2024"                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Key Code Locations Summary

| Task | File | Lines | What It Does |
|------|------|-------|--------------|
| **Receive upload** | `main.py` | 257-326 | FastAPI endpoint that receives file |
| **Parse CSV** | `main.py` | 277-278 | Reads file bytes, converts to DataFrame |
| **Analyze data** | `data_processor.py` | 32-299 | Calculates statistics from CSV |
| **Generate insights** | `insight_generator.py` | 29-436 | Converts stats to text insights |
| **Create embeddings** | `embeddings.py` | 87-112 | Converts text to vectors |
| **Store in database** | `vector_store.py` | 116-170 | Uploads to Qdrant Cloud |

---

## 🎯 Quick Reference: What Happens to Your CSV?

1. **File arrives** → `main.py` line 277: `content = await file.read()`
2. **Becomes DataFrame** → `main.py` line 278: `df = pd.read_csv(io.BytesIO(content))`
3. **Analyzed** → `data_processor.py`: Calculates click rates, templates, risk scores
4. **Becomes text** → `insight_generator.py`: "Finance department shows 32.5% click rate..."
5. **Becomes vector** → `embeddings.py`: Text → [0.123, -0.456, ...] (384 numbers)
6. **Stored in cloud** → `vector_store.py`: Uploads to Qdrant Cloud
7. **Ready for queries** → Can now answer: "What is Finance click rate?"

---

## 💡 Why This Architecture?

- **Separation of Concerns**: Each module does one thing well
- **Reusable**: Same embedding/vector store code works for all upload types
- **Scalable**: Qdrant Cloud handles millions of vectors
- **Searchable**: Embeddings enable semantic search (finds similar meanings)

---

## 🔍 Want to Modify Upload Behavior?

**To change CSV processing:**
- Edit `data_processor.py` methods

**To change insight format:**
- Edit `insight_generator.py` methods

**To change embedding model:**
- Edit `embeddings.py` line 20: `model_name = "..."`

**To change storage:**
- Edit `vector_store.py` `add_documents()` method

**To add new upload type:**
- Copy one of the upload endpoints in `main.py`
- Change collection name and processing logic




