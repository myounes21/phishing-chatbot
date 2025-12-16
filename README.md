# Phishing Campaign Analyzer & Knowledge-Aware Chatbot

A cloud-native, RAG-based analytical and informational assistant designed to analyze phishing campaign data and answer organizational knowledge questions.

## Features

*   **Phishing Campaign Analysis**: Quantitative insights into click rates, department risks, template effectiveness, and user behavior.
*   **Organizational Knowledge**: RAG-based answering of questions about company policies, services, and general security concepts.
*   **Stateless API**: FastAPI backend ready for cloud deployment (e.g., Render).
*   **Multi-Source RAG**: Combines structured data analytics (Pandas) with unstructured knowledge retrieval (Qdrant).

## Architecture

*   **Backend**: FastAPI
*   **LLM**: Groq (Llama 3)
*   **Vector Database**: Qdrant (Cloud or Local/In-Memory)
*   **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
*   **Data Processing**: Pandas

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file with your Groq API key:
    ```bash
    GROQ_API_KEY=your_groq_api_key_here
    # Optional: QDRANT_HOST=your_qdrant_host (defaults to in-memory/localhost)
    # Optional: QDRANT_PORT=6333
    ```

## Running the API

Start the FastAPI server:

```bash
uvicorn api_backend:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

*   **POST /query**: Ask a question.
    ```json
    {
      "user_query": "Which department is most vulnerable?"
    }
    ```
*   **POST /upload**: Upload a document or data file.
    *   `file`: The file to upload.
    *   `doc_type`: `phishing_data` (CSV), `org_knowledge`, or `general_knowledge`.
*   **GET /collections**: Get vector collection info.
*   **GET /health**: Health check.

## Testing

Run the test suite:

```bash
pytest test_suite.py
```
