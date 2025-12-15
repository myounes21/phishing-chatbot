# 🎣 Intelligent Phishing Campaign Analyzer

Transform complex phishing simulation data into actionable insights using AI-powered analytics.

## 🌟 Overview

This intelligent chatbot system analyzes phishing campaign results and provides security teams with instant, natural language insights. Built with **Groq LLM**, **sentence-transformers**, and **Qdrant vector database**, it combines quantitative analysis with qualitative intelligence.

### Key Features

- **📊 Quantitative Analysis**: Click rates, risk scores, response times, and statistical metrics
- **🧠 Qualitative Insights**: Behavioral patterns, psychological triggers, and contextual explanations
- **🤖 Natural Language Interface**: Ask questions in plain English, get clear answers
- **🔍 RAG-Powered**: Semantic search for relevant insights and context
- **⚡ Fast & Efficient**: Groq's high-speed LLM inference
- **🎯 Actionable Intelligence**: Get recommendations, not just data

## 🏗️ Architecture

```
CSV Data → Pandas Analysis → Insight Generation → Embeddings → Qdrant
                                                                    ↓
User Query → Groq LLM → Tool Selection → [Pandas/RAG] → Response
```

### Components

1. **Data Processor**: Pandas-based quantitative analysis
2. **Insight Generator**: Converts patterns into human-readable insights
3. **Embedding Generator**: sentence-transformers for vector creation
4. **Vector Store**: Qdrant for semantic search
5. **LLM Orchestrator**: Groq LLM for query understanding and tool selection
6. **Chatbot Interface**: Streamlit UI or CLI

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Groq API key ([Get one here](https://console.groq.com))
- Docker & Docker Compose (optional)

### Installation

#### Option 1: Local Installation

```bash
# Clone or download the project
cd phishing-chatbot

# Install dependencies
pip install -r requirements.txt

# Set your API key in .env file (one-time setup)
echo "GROQ_API_KEY=your_api_key_here" > .env

# Run Qdrant (optional - will use in-memory if skipped)
docker run -p 6333:6333 qdrant/qdrant

# Launch the Streamlit app
streamlit run chatbot_app.py
# ✨ API key auto-loads from .env - no manual entry needed!

# OR use the CLI
python chatbot_cli.py --csv sample_phishing_data.csv
```

**🎉 New Feature**: API key automatically loads from `.env` file - no need to type it every time!

#### Option 2: Docker Deployment

```bash
# Set environment variables
export GROQ_API_KEY='your-api-key-here'

# Launch everything with Docker Compose
docker-compose up -d

# Access the app at http://localhost:8501
```

## 📊 Data Format

Your CSV file should contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `User_ID` | string | Unique identifier for each user |
| `Department` | string | User's department |
| `Template` | string | Phishing template used |
| `Action` | string | User action: Clicked, Ignored, or Reported |
| `Response_Time_Sec` | integer | Time to respond (0 for ignored) |

### Example CSV

```csv
User_ID,Department,Template,Action,Response_Time_Sec
U001,Finance,Urgent Password Reset,Clicked,35
U002,Sales,CEO Impersonation,Ignored,0
U003,IT,Fake Invoice,Reported,120
```

A sample dataset is provided: `sample_phishing_data.csv`

## 💬 Usage Examples

### Streamlit Web Interface

1. Launch the app: `streamlit run chatbot_app.py`
2. Enter your Groq API key in the sidebar
3. Upload your CSV file
4. Click "Initialize System"
5. Start asking questions!

### Command-Line Interface

#### Interactive Mode
```bash
python chatbot_cli.py --csv data.csv --api-key YOUR_KEY

# Then ask questions:
🤔 You: What is the click rate for Finance?
🤔 You: Who are the top 5 riskiest users?
🤔 You: Why did the urgent template work well?
```

#### Single Query Mode
```bash
python chatbot_cli.py \
  --csv data.csv \
  --api-key YOUR_KEY \
  --query "Give me a complete risk assessment"
```

### Example Queries

**Quantitative Questions:**
- "What is the click rate for Finance?"
- "Which department has the highest click rate?"
- "Who are the top 10 riskiest users?"
- "What's the average response time?"
- "How many emails were reported?"

**Qualitative Questions:**
- "Why is Finance vulnerable to phishing?"
- "What psychological triggers does the urgent template use?"
- "How can we improve security awareness training?"
- "What behavioral patterns do high-risk users show?"

**Synthesis Questions:**
- "Give me a complete risk assessment"
- "What are the biggest security gaps?"
- "Compare departments by security awareness"
- "What templates should we focus on in training?"

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
groq:
  model: "llama-3.3-70b-versatile"  # Current recommended model
  # Alternatives: llama-3.1-70b-versatile, llama-3.1-8b-instant
  temperature: 0.1

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

qdrant:
  host: "localhost"
  port: 6333
  collection_name: "phishing_insights"

thresholds:
  high_risk_click_rate: 0.30
  fast_response_time: 60
```

## 📁 Project Structure

```
phishing-chatbot/
├── chatbot_app.py          # Streamlit web interface
├── chatbot_cli.py          # Command-line interface
├── data_processor.py       # Pandas quantitative analysis
├── insight_generator.py    # Qualitative insight generation
├── embeddings.py           # Sentence-transformers embeddings
├── vector_store.py         # Qdrant vector database
├── llm_orchestrator.py     # Groq LLM orchestration
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker deployment
├── Dockerfile              # Container definition
└── sample_phishing_data.csv # Example dataset
```

## 🛠️ Development

### Running Tests

```bash
# Test individual components
python data_processor.py
python insight_generator.py
python embeddings.py
python vector_store.py

# Test with your API key
export GROQ_API_KEY='your-key'
python llm_orchestrator.py
```

### Adding Custom Insights

Edit `insight_generator.py` to add new insight categories:

```python
def generate_custom_insights(self) -> List[Dict[str, Any]]:
    # Your custom logic here
    insights = []
    # ...
    return insights
```

### Customizing Prompts

The system prompt can be modified in `llm_orchestrator.py`:

```python
def _create_system_prompt(self) -> str:
    return """Your custom system prompt..."""
```

## 🔒 Security Considerations

- **API Keys**: Never commit API keys. Use environment variables.
- **Data Privacy**: Campaign data may contain sensitive information. Use secure storage.
- **Access Control**: Implement authentication for production deployments.
- **Data Retention**: Configure appropriate data retention policies.

## 🚀 Performance

- **Query Response Time**: < 2 seconds average
- **Embedding Generation**: ~1000 texts/second
- **Vector Search**: Sub-millisecond retrieval
- **Scalability**: Handles datasets up to 10K records efficiently

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional insight categories
- More sophisticated risk scoring
- Multi-campaign comparison
- Visualization dashboards
- Export/reporting features

## 📝 License

This project is provided as-is for educational and commercial use.

## 🙋 Support

For questions or issues:
1. Check the example queries
2. Review the configuration
3. Verify your API key
4. Check logs for errors

## 🔄 Updates

**Version 1.0.0** (Current)
- Initial release
- Groq LLM integration
- Qdrant vector database
- Streamlit & CLI interfaces
- Complete RAG pipeline

## 🎓 How It Works

### 1. Data Processing
The system loads your CSV and performs statistical analysis using Pandas:
- Click rates by department, template, user
- Response time patterns
- Risk scoring algorithms

### 2. Insight Generation
Converts numerical patterns into human-readable insights:
- Department vulnerabilities
- Template effectiveness
- User risk profiles
- Behavioral patterns

### 3. Vector Embedding
Uses sentence-transformers to create 384-dimensional embeddings:
- Semantic representation of insights
- Enables similarity-based retrieval

### 4. Vector Storage
Stores embeddings in Qdrant for fast semantic search:
- Cosine similarity matching
- Category filtering
- Relevance scoring

### 5. Query Processing
Groq LLM acts as intelligent orchestrator:
- Understands user intent
- Selects appropriate tools (Pandas/RAG)
- Synthesizes comprehensive responses

## 📊 Sample Output

```
Query: "Who are the top 3 riskiest users and why?"

Response:
Based on the analysis, here are the top 3 highest-risk users:

1. **User U025** (Finance) - Risk Score: 15.67
   - Clicked 4 phishing emails with average response time of 21 seconds
   - Extremely vulnerable to urgency-based templates
   - Shows impulsive clicking behavior without verification
   - Recommendation: Immediate one-on-one security training

2. **User U011** (Finance) - Risk Score: 12.34
   - Clicked 3 phishing emails, average response 25 seconds
   - Particularly susceptible to financial/banking themes
   - Needs training on email verification techniques

3. **User U007** (Finance) - Risk Score: 11.89
   - Clicked 3 phishing emails, fastest response at 15 seconds
   - High vulnerability to time-pressure tactics
   - Recommend implementing email verification checklist

Pattern: All top-risk users are from Finance department, suggesting 
department-wide security awareness training is critical. Focus training 
on recognizing urgency tactics and proper email verification procedures.
```

## 🎯 Use Cases

1. **Post-Campaign Analysis**: Understand campaign results
2. **Security Training**: Identify who needs training
3. **Template Testing**: Evaluate template effectiveness
4. **Risk Assessment**: Prioritize security interventions
5. **Trend Analysis**: Track improvement over time
6. **Executive Reporting**: Generate insights for leadership

## 🌐 API Integration

The system can be extended with a REST API:

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/query")
async def query(question: str):
    response = chatbot.query(question)
    return {"response": response}
```

## 📈 Roadmap

- [ ] Multi-campaign comparison
- [ ] Temporal trend analysis
- [ ] Interactive visualizations
- [ ] PDF report generation
- [ ] Scheduled analysis reports
- [ ] Integration with security platforms
- [ ] Real-time campaign monitoring
- [ ] Machine learning predictions

---

**Built with ❤️ for cybersecurity teams**

*Making phishing analysis intelligent, accessible, and actionable.*
