# 🪟 Windows Quick Start Guide

## The Error You're Seeing

```
qdrant_client.http.exceptions.ResponseHandlingException: 
[WinError 10061] No connection could be made because the target machine actively refused it
```

**This means**: Qdrant is trying to connect to a server that isn't running. Don't worry - the system now automatically falls back to in-memory storage!

---

## ✅ Solution: Use In-Memory Storage (Easiest)

The system is **already configured** to automatically use in-memory storage if Qdrant isn't running. Just run the app and it will work!

### Quick Start (No Qdrant Server Needed)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
# Option A: Environment variable
set GROQ_API_KEY=your_api_key_here

# Option B: Create .env file
echo GROQ_API_KEY=your_api_key_here > .env

# 3. Run the app
streamlit run chatbot_app.py

# OR use CLI
python chatbot_cli.py --csv sample_phishing_data.csv
```

**That's it!** The system will automatically use in-memory storage.

---

## 🐳 Option 2: Run Qdrant with Docker (For Persistent Storage)

If you want data to persist between sessions, run Qdrant:

### Prerequisites
- Install Docker Desktop for Windows from https://www.docker.com/products/docker-desktop

### Steps

```bash
# 1. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 2. Verify it's running
curl http://localhost:6333/health
# Should return: {"title":"qdrant - vector search engine","version":"..."}

# 3. Now run your app
streamlit run chatbot_app.py
```

---

## 🔧 Troubleshooting Windows Issues

### Issue 1: Python Not Found
```bash
# Make sure Python is in PATH
python --version

# If not found, download from:
# https://www.python.org/downloads/
# ✓ Check "Add Python to PATH" during installation
```

### Issue 2: pip Command Not Found
```bash
# Use python -m pip instead
python -m pip install -r requirements.txt
```

### Issue 3: Virtual Environment on Windows
```bash
# Create virtual environment
python -m venv venv

# Activate (Command Prompt)
venv\Scripts\activate.bat

# Activate (PowerShell)
venv\Scripts\Activate.ps1

# If PowerShell gives error, run first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 4: Docker Not Starting
```bash
# Make sure Docker Desktop is running
# Look for Docker icon in system tray

# Test Docker
docker --version

# If issues, restart Docker Desktop
```

### Issue 5: Port 8501 Already in Use
```bash
# Use a different port
streamlit run chatbot_app.py --server.port 8502

# Then access at: http://localhost:8502
```

---

## 📝 Complete Windows Installation

### Step-by-Step Instructions

1. **Install Python 3.9+**
   - Download: https://www.python.org/downloads/
   - ✓ Check "Add Python to PATH"
   - ✓ Check "Install pip"

2. **Download Project Files**
   - Extract all files to a folder (e.g., `C:\phishing-chatbot\`)

3. **Open Command Prompt**
   ```bash
   # Navigate to project folder
   cd C:\phishing-chatbot
   ```

4. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate.bat
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Set API Key**
   ```bash
   # Create .env file
   notepad .env
   
   # Add this line and save:
   GROQ_API_KEY=your_actual_api_key_here
   ```

7. **Run the Application**
   ```bash
   streamlit run chatbot_app.py
   ```

8. **Access in Browser**
   - Automatically opens: http://localhost:8501
   - Or manually navigate to that URL

---

## 🎯 Recommended Setup for Windows

### Without Docker (Simplest)
```bash
# This will use in-memory storage automatically
pip install -r requirements.txt
set GROQ_API_KEY=your_key
streamlit run chatbot_app.py
```

### With Docker (For Production)
```bash
# 1. Install Docker Desktop
# 2. Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3. Run app
streamlit run chatbot_app.py
```

---

## 📊 What's Different in In-Memory Mode

| Feature | Qdrant Server | In-Memory |
|---------|--------------|-----------|
| Setup | Need Docker | Automatic |
| Performance | Faster | Fast enough |
| Persistence | Data saved | Data lost on restart |
| Best for | Production | Development/Demo |

**Both work perfectly!** In-memory is fine for:
- Testing the system
- Analyzing single campaigns
- Demo purposes
- Development

Use Qdrant server when:
- Running in production
- Need data persistence
- Multiple campaigns
- High traffic

---

## 🚀 Quick Test

Run this to verify everything works:

```bash
# Test with CLI (no UI needed)
python chatbot_cli.py --csv sample_phishing_data.csv --query "What is the click rate?"
```

You should see:
```
📊 [1/6] Loading phishing campaign data...
    ✓ Loaded 50 records
...
✓ Using in-memory Qdrant storage
...
💡 Response:
[Answer about click rates]
```

---

## 💡 Pro Tips for Windows

1. **Use PowerShell or Windows Terminal** (better than CMD)
   - Install Windows Terminal from Microsoft Store
   - Much better experience

2. **Use WSL2 for Linux Experience** (Optional)
   ```bash
   # Install WSL2
   wsl --install
   
   # Then follow Linux instructions
   ```

3. **VS Code Integration**
   - Install VS Code
   - Open project folder
   - Use integrated terminal
   - Python extension helps a lot

4. **Antivirus Issues**
   - If installs fail, temporarily disable antivirus
   - Or add Python folder to exclusions

---

## ✅ Verification Checklist

- [ ] Python 3.9+ installed (`python --version`)
- [ ] pip working (`pip --version`)
- [ ] Dependencies installed (`pip list | grep groq`)
- [ ] API key set (check `.env` file exists)
- [ ] Port 8501 available
- [ ] Internet connection (for Groq API calls)

---

## 🆘 Still Having Issues?

### Check Python Installation
```bash
python --version
# Should show: Python 3.9.x or higher
```

### Check Dependencies
```bash
pip list
# Should include: groq, sentence-transformers, qdrant-client, streamlit
```

### Test Imports
```bash
python -c "import groq; print('✓ Groq OK')"
python -c "import qdrant_client; print('✓ Qdrant OK')"
python -c "import streamlit; print('✓ Streamlit OK')"
```

### View Detailed Errors
```bash
# Run with verbose logging
python chatbot_cli.py --csv sample_phishing_data.csv 2>&1 | more
```

---

## 📞 Common Error Messages

### "ModuleNotFoundError: No module named 'X'"
```bash
# Solution: Install the missing module
pip install X
```

### "Permission denied"
```bash
# Solution: Run as administrator
# Right-click Command Prompt → "Run as administrator"
```

### "streamlit: command not found"
```bash
# Solution: Use full path or python -m
python -m streamlit run chatbot_app.py
```

---

## 🎉 Success!

Once you see this, you're ready:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

**The system is now running with in-memory storage!**

---

## Next Steps

1. Upload your phishing campaign CSV
2. Enter your Groq API key
3. Click "Initialize System"
4. Start asking questions!

The in-memory storage is **perfect for getting started**. You can always add Qdrant server later if needed.
