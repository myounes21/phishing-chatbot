# 🔑 API Key Auto-Loading Feature

## What Changed

The Streamlit app now **automatically loads your API key** from the `.env` file. No need to enter it manually every time!

---

## How It Works

### Before (Old Way - Annoying 😩)
1. Start Streamlit
2. Manually type API key in sidebar
3. Every. Single. Time.

### After (New Way - Easy! 🎉)
1. Set API key in `.env` file once
2. Start Streamlit
3. API key automatically loaded! ✨

---

## Setup (One-Time)

### Step 1: Create .env File

**Option A: Using Command Line**
```bash
# Windows
echo GROQ_API_KEY=your_actual_api_key_here > .env

# Mac/Linux
echo "GROQ_API_KEY=your_actual_api_key_here" > .env
```

**Option B: Using Text Editor**
1. Create a file named `.env` in the project folder
2. Add this line:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```
3. Save the file

### Step 2: Verify It Works

```bash
# Run the app
streamlit run chatbot_app.py

# You should see in the sidebar:
# ✅ API key loaded from .env file
# Using key: gsk_abcd...xyz
```

That's it! No more manual entry.

---

## UI Changes

### When .env File Exists with API Key

**Sidebar shows:**
```
⚙️ Configuration
✅ API key loaded from .env file
Using key: gsk_1234...9999

🔧 Override API key (optional)
   [Expandable section if you want to use a different key]
```

### When .env File Doesn't Exist

**Sidebar shows:**
```
⚙️ Configuration
⚠️ No API key found in .env file

Groq API Key: [Text input box]
💡 Tip: Add to .env file to skip this step next time
```

---

## Benefits

✅ **Convenient**: Set once, use forever  
✅ **Secure**: API key not exposed in code  
✅ **Flexible**: Can still override if needed  
✅ **Professional**: Industry-standard practice  
✅ **Time-Saving**: No repetitive typing  

---

## Additional Features

### 1. Override Option
If you need to use a different API key temporarily:
1. Click "🔧 Override API key (optional)" in sidebar
2. Enter the different key
3. System uses the override key instead

### 2. Key Masking
Your API key is displayed as: `gsk_1234...9999`  
(First 8 chars + ... + last 4 chars)

### 3. Helpful Messages
- ✅ Success: "API key loaded from .env file"
- ⚠️ Warning: "No API key found in .env file"
- 💡 Tips: Guidance on what to do next

---

## Security Best Practices

### ✅ DO:
- Keep `.env` file in `.gitignore`
- Never commit `.env` to version control
- Use different keys for dev/prod
- Rotate keys periodically

### ❌ DON'T:
- Share `.env` file
- Hardcode keys in source code
- Commit keys to Git
- Use same key everywhere

---

## Troubleshooting

### "API key loaded but not working"

**Check:**
1. No spaces around the equals sign:
   ```
   # ✅ Correct
   GROQ_API_KEY=gsk_abc123
   
   # ❌ Wrong
   GROQ_API_KEY = gsk_abc123
   ```

2. No quotes needed:
   ```
   # ✅ Correct
   GROQ_API_KEY=gsk_abc123
   
   # ❌ Wrong (but might work)
   GROQ_API_KEY="gsk_abc123"
   ```

3. Key is actually valid - test it:
   ```bash
   python -c "from groq import Groq; print(Groq(api_key='YOUR_KEY').models.list())"
   ```

### ".env file exists but app asks for key"

**Solutions:**
1. Restart Streamlit (Ctrl+C and run again)
2. Check `.env` is in the same folder as `chatbot_app.py`
3. Verify file is named `.env` not `env.txt` or `.env.txt`
4. On Windows, make sure you can see file extensions

### "Cannot create .env file"

**Windows users:**
```bash
# Use PowerShell or Command Prompt, not File Explorer
# File Explorer makes it hard to create files starting with dot

# PowerShell
New-Item .env -ItemType File
Add-Content .env "GROQ_API_KEY=your_key_here"

# Or use Notepad
notepad .env
# When saving, set "Save as type" to "All Files"
```

---

## Example .env File

```bash
# Phishing Campaign Analyzer Configuration

# Required: Groq API Key
GROQ_API_KEY=gsk_your_actual_key_here

# Optional: Custom settings
# QDRANT_HOST=localhost
# QDRANT_PORT=6333
# LOG_LEVEL=INFO
```

---

## Quick Test

Verify your setup:

```bash
# 1. Check .env exists
ls -la | grep .env     # Mac/Linux
dir /a | findstr .env  # Windows

# 2. Check content (without showing key in terminal)
cat .env | grep GROQ   # Mac/Linux
type .env | findstr GROQ  # Windows

# 3. Test loading
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✓ Key loaded' if os.getenv('GROQ_API_KEY') else '✗ Not found')"

# 4. Run app
streamlit run chatbot_app.py
# Look for "✅ API key loaded from .env file" in sidebar
```

---

## Migration Guide

### If You Were Using Environment Variables

```bash
# Old way (still works!)
export GROQ_API_KEY=your_key
streamlit run chatbot_app.py

# New way (better!)
echo "GROQ_API_KEY=your_key" > .env
streamlit run chatbot_app.py
```

Both work! The app checks:
1. Environment variables first
2. Then `.env` file
3. Then asks manually

---

## Pro Tips

### 1. Multiple Environments
```bash
# Development
.env.development

# Production  
.env.production

# Load specific one:
cp .env.development .env
streamlit run chatbot_app.py
```

### 2. Template File
Keep `.env.template` in version control:
```bash
# .env.template (safe to commit)
GROQ_API_KEY=your_key_here
QDRANT_HOST=localhost

# .env (never commit!)
GROQ_API_KEY=gsk_actual_secret_key
QDRANT_HOST=localhost
```

### 3. Team Setup
```bash
# In your README.md
## Setup
1. Copy `.env.template` to `.env`
2. Add your API key to `.env`
3. Run `streamlit run chatbot_app.py`
```

---

## Summary

**Old workflow:**
```
Start app → Type API key → Use app → Close app
Next time → Type API key again → Use app → 😩
```

**New workflow:**
```
Create .env once → Start app → Use app → Close app
Next time → Start app → Use app → 😊
```

**Much better!** 🎉

---

## Related Files

- **chatbot_app.py**: Main app with auto-loading logic
- **requirements.txt**: Includes `python-dotenv>=1.0.0`
- **.env.template**: Template file (copy to `.env`)
- **WINDOWS_GUIDE.md**: Windows-specific setup help

---

## Need Help?

If API key auto-loading isn't working:
1. Check **TROUBLESHOOTING.md** → "Cannot connect to Groq API"
2. Verify `.env` file location
3. Restart Streamlit completely
4. Test with manual entry to confirm key works
