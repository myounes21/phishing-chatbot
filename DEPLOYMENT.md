# 🚀 Deployment Guide

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Considerations](#production-considerations)
5. [Monitoring & Logging](#monitoring--logging)
6. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites
- Python 3.9+
- pip
- Git
- (Optional) Docker

### Setup Steps

1. **Clone/Download the Project**
```bash
cd phishing-chatbot
```

2. **Run Setup Script**
```bash
chmod +x setup.sh
./setup.sh
```

3. **Configure Environment**
```bash
# Edit .env file
nano .env

# Add your Groq API key:
GROQ_API_KEY=your_actual_api_key_here
```

4. **Start Qdrant (Optional)**
```bash
# Option 1: Docker
docker run -p 6333:6333 qdrant/qdrant

# Option 2: Skip this step - system will use in-memory storage
```

5. **Run the Application**
```bash
# Activate virtual environment
source venv/bin/activate

# Start Streamlit
streamlit run chatbot_app.py

# OR use CLI
python chatbot_cli.py --csv sample_phishing_data.csv
```

---

## Docker Deployment

### Quick Start

1. **Set Environment Variables**
```bash
export GROQ_API_KEY='your_api_key_here'
```

2. **Launch with Docker Compose**
```bash
docker-compose up -d
```

3. **Access the Application**
- Streamlit UI: http://localhost:8501
- Qdrant Dashboard: http://localhost:6333/dashboard

4. **View Logs**
```bash
docker-compose logs -f chatbot
```

5. **Stop Services**
```bash
docker-compose down
```

### Custom Docker Build

```bash
# Build image
docker build -t phishing-chatbot .

# Run container
docker run -d \
  -p 8501:8501 \
  -e GROQ_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  --name phishing-chatbot \
  phishing-chatbot
```

---

## Cloud Deployment

### AWS Deployment

#### Option 1: EC2 Instance

```bash
# 1. Launch EC2 instance (Ubuntu 22.04, t3.medium or larger)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv git docker.io docker-compose

# 4. Clone project
git clone your-repo-url
cd phishing-chatbot

# 5. Run setup
./setup.sh

# 6. Configure .env with API keys

# 7. Start services
docker-compose up -d

# 8. Configure security group
# - Allow inbound TCP 8501 (Streamlit)
# - Allow inbound TCP 6333 (Qdrant, optional)
```

#### Option 2: ECS (Elastic Container Service)

```yaml
# task-definition.json
{
  "family": "phishing-chatbot",
  "containerDefinitions": [
    {
      "name": "qdrant",
      "image": "qdrant/qdrant:latest",
      "memory": 1024,
      "portMappings": [{"containerPort": 6333}]
    },
    {
      "name": "chatbot",
      "image": "your-ecr-repo/phishing-chatbot:latest",
      "memory": 2048,
      "portMappings": [{"containerPort": 8501}],
      "environment": [
        {"name": "GROQ_API_KEY", "value": "your_key"},
        {"name": "QDRANT_HOST", "value": "localhost"}
      ]
    }
  ]
}
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# 1. Build and push image
gcloud builds submit --tag gcr.io/your-project/phishing-chatbot

# 2. Deploy
gcloud run deploy phishing-chatbot \
  --image gcr.io/your-project/phishing-chatbot \
  --platform managed \
  --region us-central1 \
  --set-env-vars GROQ_API_KEY=your_key \
  --allow-unauthenticated \
  --memory 2Gi
```

### Azure Deployment

#### Container Instances

```bash
# 1. Create resource group
az group create --name phishing-rg --location eastus

# 2. Create container
az container create \
  --resource-group phishing-rg \
  --name phishing-chatbot \
  --image your-registry/phishing-chatbot:latest \
  --dns-name-label phishing-analyzer \
  --ports 8501 \
  --environment-variables GROQ_API_KEY=your_key
```

### Heroku Deployment

```bash
# 1. Create Procfile
echo "web: streamlit run chatbot_app.py --server.port=$PORT" > Procfile

# 2. Create heroku.yml
cat > heroku.yml << EOF
build:
  docker:
    web: Dockerfile
EOF

# 3. Deploy
heroku create your-app-name
heroku stack:set container
heroku config:set GROQ_API_KEY=your_key
git push heroku main
```

---

## Production Considerations

### Security

1. **API Key Management**
```bash
# Use secrets management
# AWS: AWS Secrets Manager
# GCP: Secret Manager
# Azure: Key Vault

# Example with AWS:
aws secretsmanager create-secret \
  --name phishing-chatbot-groq-key \
  --secret-string "your_api_key"
```

2. **Access Control**
```python
# Add authentication to Streamlit app
# In chatbot_app.py, add:

import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    'phishing_analyzer',
    'auth_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show app
    main()
elif authentication_status == False:
    st.error('Username/password is incorrect')
```

3. **HTTPS/TLS**
```bash
# Use reverse proxy (nginx)
# /etc/nginx/sites-available/phishing-chatbot

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Scalability

1. **Horizontal Scaling**
```yaml
# docker-compose.yml for multiple instances
version: '3.8'
services:
  chatbot:
    image: phishing-chatbot
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G
    # ... rest of config
```

2. **Load Balancing**
```nginx
# nginx load balancer config
upstream chatbot_backend {
    least_conn;
    server chatbot1:8501;
    server chatbot2:8501;
    server chatbot3:8501;
}

server {
    location / {
        proxy_pass http://chatbot_backend;
    }
}
```

3. **Caching**
```python
# Add caching to expensive operations
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(query_hash):
    return orchestrator.process_query(query)
```

### Performance Optimization

1. **Connection Pooling**
```python
# For Qdrant connections
from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True  # Faster than REST
)
```

2. **Batch Processing**
```python
# Process multiple queries in batch
def batch_process_queries(queries):
    embeddings = embedding_gen.encode_text(queries)
    results = []
    for query, emb in zip(queries, embeddings):
        result = vector_store.search(emb.tolist())
        results.append(result)
    return results
```

3. **Model Optimization**
```python
# Use quantized models for faster inference
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device='cuda',  # Use GPU if available
    model_kwargs={'torch_dtype': torch.float16}  # Half precision
)
```

---

## Monitoring & Logging

### Application Monitoring

1. **Structured Logging**
```python
# logging_config.py
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        }
        return json.dumps(log_obj)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(handlers=[handler], level=logging.INFO)
```

2. **Metrics Collection**
```python
# Add to chatbot_app.py
from prometheus_client import Counter, Histogram

query_counter = Counter('queries_total', 'Total queries processed')
query_duration = Histogram('query_duration_seconds', 'Query processing time')

@query_duration.time()
def process_query(query):
    query_counter.inc()
    # ... process query
```

3. **Health Checks**
```python
# health_check.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    checks = {
        'database': check_qdrant_connection(),
        'llm': check_groq_connection(),
        'embeddings': check_model_loaded()
    }
    return {
        'status': 'healthy' if all(checks.values()) else 'unhealthy',
        'checks': checks
    }
```

### Log Aggregation

**Using ELK Stack:**
```yaml
# docker-compose.yml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
  
  logstash:
    image: docker.elastic.co/logstash/logstash:8.0.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
  
  kibana:
    image: docker.elastic.co/kibana/kibana:8.0.0
    ports:
      - "5601:5601"
```

---

## Troubleshooting

### Common Issues

#### 1. Groq API Connection Errors
```bash
# Check API key
echo $GROQ_API_KEY

# Test connection
curl -X POST https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mixtral-8x7b-32768","messages":[{"role":"user","content":"test"}]}'
```

#### 2. Qdrant Connection Failed
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Check logs
docker logs qdrant_container

# Test connection
curl http://localhost:6333/health
```

#### 3. Out of Memory Errors
```bash
# Increase Docker memory limit
docker update --memory 4g phishing-chatbot

# Or in docker-compose.yml:
services:
  chatbot:
    mem_limit: 4g
```

#### 4. Slow Query Performance
```python
# Add query caching
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def get_cached_response(query_hash):
    return orchestrator.process_query(query)

# Use it
query_hash = hashlib.md5(query.encode()).hexdigest()
response = get_cached_response(query_hash)
```

### Debug Mode

```bash
# Run in debug mode
export LOG_LEVEL=DEBUG
streamlit run chatbot_app.py --logger.level=debug

# Check logs
tail -f logs/chatbot.log
```

### Performance Profiling

```python
# Add profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
response = orchestrator.process_query(query)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)
```

---

## Backup & Recovery

### Data Backup

```bash
# Backup Qdrant data
docker exec qdrant tar czf /tmp/qdrant_backup.tar.gz /qdrant/storage
docker cp qdrant:/tmp/qdrant_backup.tar.gz ./backups/

# Backup application data
tar czf data_backup_$(date +%Y%m%d).tar.gz data/
```

### Disaster Recovery

```bash
# Restore from backup
docker cp ./backups/qdrant_backup.tar.gz qdrant:/tmp/
docker exec qdrant tar xzf /tmp/qdrant_backup.tar.gz -C /
docker restart qdrant
```

---

## Maintenance

### Regular Updates

```bash
# Update dependencies
pip list --outdated
pip install -U -r requirements.txt

# Update Docker images
docker-compose pull
docker-compose up -d
```

### Database Maintenance

```bash
# Optimize Qdrant collection
curl -X POST http://localhost:6333/collections/phishing_insights/optimize
```

---

For additional help, see:
- README.md
- API documentation at /docs (if using FastAPI)
- GitHub Issues (if open source)
