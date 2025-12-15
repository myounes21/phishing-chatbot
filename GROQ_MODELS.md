# 🔄 Groq Model Update Guide

## ⚠️ Important: Mixtral Model Deprecated

The `mixtral-8x7b-32768` model has been **decommissioned** by Groq and is no longer available.

## ✅ What Was Fixed

All files have been updated to use: **`llama-3.3-70b-versatile`**

This is Groq's current recommended model (as of December 2024).

---

## 🆕 Current Supported Models

### Recommended: Llama 3.3 70B Versatile
```yaml
model: "llama-3.3-70b-versatile"
```
- ✅ **Best performance** - newest model
- ✅ **70B parameters** - high capability
- ✅ **Great reasoning** - complex queries
- ✅ **Context length**: 128K tokens
- 🎯 **Use for**: Production, complex analysis

### Alternative: Llama 3.1 70B Versatile
```yaml
model: "llama-3.1-70b-versatile"
```
- ✅ **Stable and proven**
- ✅ **70B parameters**
- ✅ **Reliable performance**
- ✅ **Context length**: 128K tokens
- 🎯 **Use for**: When you want maximum stability

### Fast Option: Llama 3.1 8B Instant
```yaml
model: "llama-3.1-8b-instant"
```
- ⚡ **Very fast** - instant responses
- ⚠️ **8B parameters** - less capable
- ✅ **Good for simple queries**
- ✅ **Context length**: 128K tokens
- 🎯 **Use for**: Development, testing, simple analysis

---

## 🔧 How to Change Models

### Option 1: Edit config.yaml (Recommended)
```yaml
# config.yaml
groq:
  model: "llama-3.3-70b-versatile"  # Change this line
```

### Option 2: Edit .env File
```bash
# .env
GROQ_MODEL=llama-3.3-70b-versatile
```

### Option 3: CLI Argument
```bash
python chatbot_cli.py \
  --csv data.csv \
  --model llama-3.1-8b-instant
```

### Option 4: In Code
```python
orchestrator = LLMOrchestrator(
    api_key=api_key,
    model="llama-3.3-70b-versatile"
)
```

---

## 📊 Model Comparison

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| **llama-3.3-70b-versatile** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Production |
| llama-3.1-70b-versatile | ⭐⭐⭐ | ⭐⭐⭐⭐ | 💰💰💰 | Stable prod |
| llama-3.1-8b-instant | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 💰 | Dev/Testing |

---

## 🚨 Error Messages You Might See

### "Model Decommissioned" Error
```
Error code: 400 - The model `mixtral-8x7b-32768` has been decommissioned 
and is no longer supported.
```

**Solution**: Update to current model (already done in latest files!)

### "Model Not Found" Error
```
Error code: 404 - The model `xyz` does not exist
```

**Solution**: Check model name spelling. Valid models:
- `llama-3.3-70b-versatile`
- `llama-3.1-70b-versatile`
- `llama-3.1-8b-instant`

### "Invalid Request" Error
```
Error code: 400 - Invalid request
```

**Solution**: Verify your API key is valid and has access to the model.

---

## 📋 Files Updated

All these files now use `llama-3.3-70b-versatile`:

1. ✅ **config.yaml** - Default configuration
2. ✅ **llm_orchestrator.py** - LLM class default
3. ✅ **chatbot_cli.py** - CLI default argument
4. ✅ **.env.template** - Environment template
5. ✅ **README.md** - Documentation examples

**No action needed** - Just download the latest files!

---

## 🔍 Check Current Available Models

To see what models are currently available:

```python
from groq import Groq
import os

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
models = client.models.list()

print("Available models:")
for model in models.data:
    print(f"  - {model.id}")
```

Or via API:
```bash
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"
```

---

## 💡 Pro Tips

### 1. Use Fastest Model for Development
```yaml
# config.yaml for dev environment
groq:
  model: "llama-3.1-8b-instant"  # Fast iterations
```

### 2. Use Best Model for Production
```yaml
# config.yaml for production
groq:
  model: "llama-3.3-70b-versatile"  # Best quality
```

### 3. Environment-Based Selection
```python
import os

# Automatically choose based on environment
if os.getenv('ENV') == 'production':
    model = "llama-3.3-70b-versatile"
else:
    model = "llama-3.1-8b-instant"
```

### 4. Fallback Strategy
```python
try:
    # Try newest model first
    orchestrator = LLMOrchestrator(model="llama-3.3-70b-versatile")
except Exception:
    # Fallback to stable model
    orchestrator = LLMOrchestrator(model="llama-3.1-70b-versatile")
```

---

## 🎯 Migration Checklist

If you have old code using Mixtral:

- [ ] Update `config.yaml` model name
- [ ] Update `.env` GROQ_MODEL variable
- [ ] Update any hardcoded model names in custom code
- [ ] Test with new model
- [ ] Monitor performance/quality differences
- [ ] Update documentation/comments

---

## 📊 Performance Expectations

### Query Response Times (Approximate)

**Simple Query**: "What is the click rate?"
- llama-3.3-70b: ~1-2 seconds
- llama-3.1-70b: ~1-2 seconds  
- llama-3.1-8b: ~0.5-1 second

**Complex Query**: "Analyze all departments and provide recommendations"
- llama-3.3-70b: ~3-5 seconds
- llama-3.1-70b: ~3-5 seconds
- llama-3.1-8b: ~1-2 seconds

**Note**: Quality of 8B model may be lower for complex analysis.

---

## 🔮 Future-Proofing

### Stay Updated with Model Changes

1. **Check Groq Docs Regularly**
   - https://console.groq.com/docs/models

2. **Monitor Deprecation Notices**
   - https://console.groq.com/docs/deprecations

3. **Use Environment Variables**
   - Makes updates easier
   - No code changes needed

4. **Test New Models**
   - Groq releases new models regularly
   - Test for quality/performance improvements

---

## 🆘 Troubleshooting

### "Still getting Mixtral error"

**Reason**: Using old version of files

**Solution**:
```bash
# Download latest files again
# Or manually update model name in config.yaml:
nano config.yaml  # Change model line
```

### "New model gives different responses"

**Normal**: Different models have different characteristics

**Solutions**:
- Adjust temperature (higher = more creative, lower = more focused)
- Modify system prompts if needed
- Test and tune for your use case

### "8B model not good enough"

**Expected**: 8B has fewer parameters than 70B

**Solution**: Use 70B models for production:
```yaml
model: "llama-3.3-70b-versatile"
```

---

## 📚 Additional Resources

- **Groq Models Documentation**: https://console.groq.com/docs/models
- **Model Deprecations**: https://console.groq.com/docs/deprecations
- **API Reference**: https://console.groq.com/docs/api-reference
- **Model Benchmarks**: https://console.groq.com/docs/benchmarks

---

## ✅ Summary

**What changed**: 
- ❌ OLD: `mixtral-8x7b-32768` (deprecated)
- ✅ NEW: `llama-3.3-70b-versatile` (current)

**What you need to do**:
- ✅ Use updated files (already done!)
- ✅ No code changes needed
- ✅ Just run the app

**The system now works with current Groq models!** 🎉
