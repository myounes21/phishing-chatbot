# 🔧 Troubleshooting Guide - Connection Issues

## Error: WinError 10061 - Connection Refused

### What This Means
The application tried to connect to Qdrant vector database at `localhost:6333`, but nothing is listening on that port.

### ✅ SOLUTION: System Now Auto-Handles This!

**Good news**: The latest version automatically falls back to in-memory storage. You don't need to do anything!

---

## Understanding the Fix

### What Changed
```python
# OLD: Would crash if Qdrant not running
client = QdrantClient(host="localhost", port=6333)

# NEW: Automatically tries in-memory if server not available
try:
    client = QdrantClient(host="localhost", port=6333, timeout=5)
except:
    client = QdrantClient(":memory:")  # Automatic fallback!
```

### What You'll See
```
⚠ Could not connect to Qdrant at localhost:6333
ℹ Falling back to in-memory storage...
✓ Using in-memory Qdrant storage
  Note: Data will not persist after restart
  To use persistent storage, start Qdrant with:
  docker run -p 6333:6333 qdrant/qdrant
```

**This is normal and expected!** The system works perfectly in this mode.

---

## When to Use Each Mode

### In-Memory Mode (Default - No Setup)
**Use when:**
- ✓ Testing the system
- ✓ Analyzing single campaigns
- ✓ Demo or presentation
- ✓ Development work
- ✓ Don't have Docker installed

**Characteristics:**
- ✓ No setup required
- ✓ Works immediately
- ✓ Fast performance
- ⚠ Data lost when app restarts
- ⚠ Can't share data between instances

### Server Mode (Optional - Requires Docker)
**Use when:**
- ✓ Production deployment
- ✓ Need data persistence
- ✓ Multiple users/instances
- ✓ Long-running analysis
- ✓ Building analytics over time

**Characteristics:**
- ✓ Data persists
- ✓ Can share between instances
- ✓ Production-ready
- ⚠ Requires Docker setup
- ⚠ Additional resource usage

---

## How to Enable Persistent Storage (Optional)

### On Windows

#### Step 1: Install Docker Desktop
1. Download: https://www.docker.com/products/docker-desktop
2. Install and restart computer
3. Start Docker Desktop
4. Wait for Docker to be running (green icon in system tray)

#### Step 2: Start Qdrant
```bash
# In Command Prompt or PowerShell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

#### Step 3: Verify It's Running
```bash
# Check Docker
docker ps

# Should show:
# CONTAINER ID   IMAGE            PORTS                    
# xxxxx          qdrant/qdrant    0.0.0.0:6333->6333/tcp

# Test connection
curl http://localhost:6333/health
# OR open in browser: http://localhost:6333/dashboard
```

#### Step 4: Run Your App
```bash
streamlit run chatbot_app.py
```

Now it will use persistent storage!

### On Mac/Linux

```bash
# Install Docker (if not installed)
# Mac: Download Docker Desktop
# Linux: sudo apt install docker.io

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Verify
curl http://localhost:6333/health

# Run app
streamlit run chatbot_app.py
```

---

## Other Connection Issues

### Issue 1: "Cannot connect to Groq API" or "Model Decommissioned"

**Symptoms:**
```
Error: 401 Unauthorized
OR
Error: API key is invalid
OR
Error code: 400 - The model `mixtral-8x7b-32768` has been decommissioned
```

**Solutions:**

**For "Model Decommissioned" Error:**
1. **Already Fixed!** Latest files use current model: `llama-3.3-70b-versatile`
2. If using old files, update config.yaml:
   ```yaml
   groq:
     model: "llama-3.3-70b-versatile"
   ```
3. See **GROQ_MODELS.md** for complete model guide

**For API Key Errors:**
1. Check your API key is correct
   ```bash
   # Windows
   echo %GROQ_API_KEY%
   
   # Mac/Linux
   echo $GROQ_API_KEY
   ```

2. Verify .env file
   ```bash
   # Open .env file
   notepad .env  # Windows
   nano .env     # Mac/Linux
   
   # Should contain:
   GROQ_API_KEY=your_actual_key_here
   ```

3. Get a new key: https://console.groq.com

4. Test the key:
   ```python
   python -c "from groq import Groq; print(Groq(api_key='YOUR_KEY').models.list())"
   ```

### Issue 2: "Port 8501 already in use"

**Symptoms:**
```
OSError: [Errno 98] Address already in use
```

**Solutions:**

**Option A: Use different port**
```bash
streamlit run chatbot_app.py --server.port 8502
# Access at: http://localhost:8502
```

**Option B: Kill existing process**
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F

# Mac/Linux
lsof -ti:8501 | xargs kill -9
```

### Issue 3: "Cannot find sentence-transformers models"

**Symptoms:**
```
OSError: Can't load model 'sentence-transformers/all-MiniLM-L6-v2'
```

**Solutions:**
1. Check internet connection (downloads model on first run)

2. Set cache directory:
   ```bash
   # Windows
   set TRANSFORMERS_CACHE=C:\temp\transformers_cache
   
   # Mac/Linux
   export TRANSFORMERS_CACHE=/tmp/transformers_cache
   ```

3. Download manually:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
   ```

### Issue 4: "SSL Certificate Error"

**Symptoms:**
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solutions:**
1. Update certificates:
   ```bash
   pip install --upgrade certifi
   ```

2. If behind corporate proxy:
   ```bash
   # Set proxy
   set HTTPS_PROXY=http://proxy.company.com:8080
   ```

3. Temporary workaround (not recommended for production):
   ```python
   import ssl
   ssl._create_default_https_context = ssl._create_unverified_context
   ```

---

## Diagnostic Commands

### Check System Status

```bash
# 1. Python version
python --version
# Should be 3.9+

# 2. Installed packages
pip list | grep -E "groq|qdrant|streamlit|sentence-transformers"

# 3. Test imports
python -c "import groq, qdrant_client, streamlit; print('✓ All OK')"

# 4. Check ports
netstat -an | grep -E "8501|6333"

# 5. Check environment variables
# Windows
set | findstr GROQ
# Mac/Linux
env | grep GROQ
```

### Run in Debug Mode

```bash
# Set debug logging
# Windows
set LOG_LEVEL=DEBUG

# Mac/Linux
export LOG_LEVEL=DEBUG

# Run with verbose output
python chatbot_cli.py --csv sample_phishing_data.csv 2>&1 | tee debug.log
```

---

## Network Configuration

### Firewall Issues

If you can't access the UI:

**Windows Firewall:**
1. Open "Windows Defender Firewall"
2. Click "Allow an app or feature"
3. Click "Change settings"
4. Find Python → Allow Private/Public
5. Restart application

**Corporate Firewall:**
- May block ports 6333, 8501
- Use different ports or contact IT

### Proxy Configuration

If behind a proxy:

```bash
# Windows
set HTTP_PROXY=http://proxy.company.com:8080
set HTTPS_PROXY=http://proxy.company.com:8080

# Mac/Linux
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# In Python code, add to requests:
proxies = {
    'http': 'http://proxy.company.com:8080',
    'https': 'http://proxy.company.com:8080',
}
```

---

## Performance Issues

### Slow Initialization

**Symptoms:**
- Takes 2+ minutes to start
- "Downloading models..." message

**Cause:**
- First-time download of sentence-transformers model (~90MB)

**Solutions:**
1. Wait for initial download (only once)
2. Subsequent starts will be fast
3. Use faster model (edit config.yaml):
   ```yaml
   embeddings:
     model_name: "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Smaller
   ```

### Slow Queries

**Symptoms:**
- Queries take 10+ seconds

**Solutions:**
1. Check internet connection (Groq API calls)
2. Reduce dataset size for testing
3. Use faster Groq model:
   ```python
   GROQ_MODEL=mixtral-8x7b-32768  # Faster
   # vs
   GROQ_MODEL=llama-3.1-70b-versatile  # Slower but better
   ```

---

## Data Issues

### CSV Format Errors

**Symptoms:**
```
KeyError: 'User_ID'
OR
ValueError: Missing required columns
```

**Solutions:**
1. Check CSV has correct columns:
   - User_ID
   - Department
   - Template
   - Action
   - Response_Time_Sec

2. Verify no extra spaces in headers

3. Test with sample data:
   ```bash
   python chatbot_cli.py --csv sample_phishing_data.csv
   ```

### Encoding Issues

**Symptoms:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions:**
1. Save CSV as UTF-8:
   - Excel: Save As → CSV UTF-8
   - Notepad: Save As → Encoding: UTF-8

2. Or specify encoding in code:
   ```python
   pd.read_csv(file, encoding='latin-1')
   ```

---

## Getting Help

### Before Asking for Help

Run diagnostics and gather info:

```bash
# 1. System info
python --version
pip list

# 2. Error message (full output)
python chatbot_cli.py --csv sample_phishing_data.csv 2>&1 > error.log

# 3. Environment check
# Windows
set > env.txt
# Mac/Linux
env > env.txt

# 4. Network check
ping google.com
curl https://api.groq.com/openai/v1/models
```

### Share This Info:
- Operating System (Windows 10/11, Mac, Linux)
- Python version
- Error message (full stack trace)
- What you were trying to do
- Environment variables (redact API key!)

---

## Quick Reset

If everything is broken:

```bash
# 1. Clean install
pip uninstall -y groq qdrant-client streamlit sentence-transformers
pip install -r requirements.txt

# 2. Clear cache
# Windows
rmdir /s %USERPROFILE%\.cache\huggingface
# Mac/Linux
rm -rf ~/.cache/huggingface

# 3. Reset config
cp .env.template .env
# Edit .env and add your API key

# 4. Test with sample
python chatbot_cli.py --csv sample_phishing_data.csv --query "test"
```

---

## Summary: Connection Refused Error

**You don't need to fix anything!** The system now automatically handles this.

**Just run the app:**
```bash
streamlit run chatbot_app.py
```

It will work with in-memory storage. Add Docker/Qdrant later only if you need persistence.
