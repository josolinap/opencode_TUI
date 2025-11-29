# LLM Integration: Groq (Free Tier)

## Provider Details

- **Provider**: Groq
- **API Type**: api_key
- **Models**: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b-32768
- **Requirements**: groq
- **Integration Date**: 2025-11-29T10:51:45.000000

## Description

Groq provides ultra-fast inference with free tier access to popular models. Known for exceptional speed and low latency.

## Free Tier Models

- `llama-3.1-8b-instant` - Fast, efficient 8B model
- `llama-3.1-70b-versatile` - Powerful 70B model
- `mixtral-8x7b-32768` - MoE model with 32K context
- `gemma-7b-it` - Google's Gemma model
- `gemma2-9b-it` - Latest Gemma 2 model

## Setup Instructions

```bash
# Install requirements
pip install groq

# Get free API key from https://console.groq.com/keys
export GROQ_API_KEY="your-api-key"
```

## OpenCode Configuration

```json
{
  "providers": {
    "groq": {
      "base_url": "https://api.groq.com/openai/v1",
      "api_key": "${GROQ_API_KEY}",
      "models": {
        "llama-3.1-8b": "llama-3.1-8b-instant",
        "llama-3.1-70b": "llama-3.1-70b-versatile",
        "mixtral-8x7b": "mixtral-8x7b-32768",
        "gemma-7b": "gemma-7b-it",
        "gemma2-9b": "gemma2-9b-it"
      }
    }
  }
}
```

## Usage Example

```python
from groq import Groq

client = Groq(api_key="your-groq-api-key")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Status

- [x] Documentation created
- [ ] Requirements installed (✓)
- [ ] API keys configured
- [ ] Basic functionality tested
- [ ] Integration with Neo-Clone completed

## Notes

- Groq offers generous free tier limits
- Known for fastest inference speeds
- OpenAI-compatible API
- Excellent for real-time applications

## Resilience Features

- Automatic retry on installation failures (up to 5 attempts)
- Graceful degradation when requirements unavailable
- Documentation preserved for manual integration
- Never fails - always creates some form of documentation
