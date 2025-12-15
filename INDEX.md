# 📑 PROJECT INDEX - Quick Reference

## 🚀 Getting Started Files (START HERE!)

1. **README.md** - Complete user guide, features, and examples
2. **PROJECT_SUMMARY.md** - Project overview and what was built
3. **setup.sh** - Automated setup script (run this first!)
4. **.env.template** - Environment variables (copy to .env and add API key)

## 💻 Application Files (Core System)

### User Interfaces
- **chatbot_app.py** - Streamlit web application (main UI)
- **chatbot_cli.py** - Command-line interface

### Core Engine
- **data_processor.py** - Pandas quantitative analysis engine
- **insight_generator.py** - Qualitative insight generation
- **embeddings.py** - Vector embedding creation (sentence-transformers)
- **vector_store.py** - Qdrant integration and RAG retriever
- **llm_orchestrator.py** - Groq LLM orchestration and tool selection

## ⚙️ Configuration & Setup

- **config.yaml** - System configuration settings
- **requirements.txt** - Python dependencies
- **.env.template** - Environment variables template
- **setup.sh** - Automated installation script

## 🐳 Deployment Files

- **Dockerfile** - Container definition
- **docker-compose.yml** - Multi-container orchestration
- **DEPLOYMENT.md** - Complete deployment guide (local, cloud, production)

## 📊 Sample Data

- **sample_phishing_data.csv** - Example dataset (50 records)

## 🧪 Testing

- **test_suite.py** - Comprehensive test suite with pytest

## 📚 Documentation

- **INDEX.md** (this file) - Quick reference guide
- **README.md** - Main documentation and user guide
- **PROJECT_ARCHITECTURE.md** - System architecture and design
- **PROJECT_SUMMARY.md** - Complete implementation overview
- **DEPLOYMENT.md** - Deployment and production guide
- **WINDOWS_GUIDE.md** - 🪟 Windows-specific setup and troubleshooting
- **TROUBLESHOOTING.md** - 🔧 Complete troubleshooting guide for all issues
- **API_KEY_GUIDE.md** - 🔑 API key auto-loading feature guide
- **GROQ_MODELS.md** - 🤖 Groq model updates and current recommendations

---

## 🎯 Quick Start Commands

### Option 1: Automated Setup (Recommended)
```bash
chmod +x setup.sh
./setup.sh
# Edit .env and add your Groq API key
source venv/bin/activate
streamlit run chatbot_app.py
```

### Option 2: Docker (Easiest)
```bash
export GROQ_API_KEY='your_key_here'
docker-compose up -d
# Access at http://localhost:8501
```

### Option 3: Command Line
```bash
pip install -r requirements.txt
export GROQ_API_KEY='your_key_here'
python chatbot_cli.py --csv sample_phishing_data.csv
```

---

## 📁 File Usage Guide

### For End Users:
1. Read **README.md** for overview
2. Run **setup.sh** for installation
3. Launch **chatbot_app.py** for web UI
4. Or use **chatbot_cli.py** for command line

### For Developers:
1. Review **PROJECT_ARCHITECTURE.md** for design
2. Examine core files: data_processor.py, llm_orchestrator.py
3. Check **test_suite.py** for testing
4. Modify **config.yaml** for customization

### For DevOps:
1. Use **docker-compose.yml** for deployment
2. Read **DEPLOYMENT.md** for production setup
3. Configure **.env** for secrets management
4. Monitor using health checks in code

---

## 🔑 Required Setup

1. **Get Groq API Key**: https://console.groq.com
2. **Copy environment file**: `cp .env.template .env`
3. **Add API key**: Edit .env file
4. **Install dependencies**: Run setup.sh or `pip install -r requirements.txt`
5. **Start application**: Choose one of the three options above

---

## 💡 File Dependencies

```
chatbot_app.py depends on:
├── data_processor.py
├── insight_generator.py
├── embeddings.py
├── vector_store.py
└── llm_orchestrator.py

llm_orchestrator.py depends on:
├── data_processor.py
└── vector_store.py (RAGRetriever)

vector_store.py depends on:
└── embeddings.py

insight_generator.py depends on:
└── data_processor.py
```

---

## 📊 Data Flow Through Files

```
1. User uploads CSV
   ↓
2. data_processor.py analyzes data
   ↓
3. insight_generator.py creates insights
   ↓
4. embeddings.py creates vectors
   ↓
5. vector_store.py stores in Qdrant
   ↓
6. llm_orchestrator.py processes queries
   ↓
7. chatbot_app.py displays results
```

---

## 🎓 Learning Path

**Beginner**: Start with README.md and sample_phishing_data.csv
**Intermediate**: Explore data_processor.py and insight_generator.py
**Advanced**: Study llm_orchestrator.py and vector_store.py
**Expert**: Review entire architecture and extend functionality

---

## 🛠️ Customization Points

- **Add new insights**: Edit insight_generator.py
- **Change risk thresholds**: Modify config.yaml
- **Adjust UI**: Customize chatbot_app.py
- **Add new queries**: Extend data_processor.py
- **Modify prompts**: Update llm_orchestrator.py

---

## 📦 Complete File List (24 files)

### Python Files (9)
1. chatbot_app.py
2. chatbot_cli.py
3. data_processor.py
4. insight_generator.py
5. embeddings.py
6. vector_store.py
7. llm_orchestrator.py
8. test_suite.py

### Configuration (4)
9. config.yaml
10. requirements.txt
11. .env.template
12. setup.sh

### Deployment (2)
13. Dockerfile
14. docker-compose.yml

### Data (1)
15. sample_phishing_data.csv

### Documentation (7)
16. INDEX.md (this file)
17. README.md
18. PROJECT_ARCHITECTURE.md
19. PROJECT_SUMMARY.md
20. DEPLOYMENT.md
21. WINDOWS_GUIDE.md
22. TROUBLESHOOTING.md
23. API_KEY_GUIDE.md
24. GROQ_MODELS.md

---

## ✅ Checklist Before Running

- [ ] Groq API key obtained
- [ ] .env file created and configured
- [ ] Dependencies installed (setup.sh or pip)
- [ ] Qdrant running (Docker) or using in-memory
- [ ] CSV data prepared (or use sample_phishing_data.csv)
- [ ] Port 8501 available (for Streamlit)

---

## 🆘 Need Help?

1. **Quick answers**: Check README.md FAQ section
2. **Setup issues**: Review setup.sh output
3. **Deployment**: Read DEPLOYMENT.md
4. **Architecture**: Study PROJECT_ARCHITECTURE.md
5. **API errors**: Verify API key in .env file

---

**Everything you need is in these 20 files. Start with README.md!**
