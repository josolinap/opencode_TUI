# LLM Integration: Google Gemini (Free Tier)

## Provider Details

- **Provider**: Google Gemini
- **API Type**: api_key
- **Models**: gemini-1.5-flash, gemini-1.5-pro, gemini-pro
- **Requirements**: google-generativeai
- **Integration Date**: 2025-11-29T10:52:00.000000

## Description

Google Gemini models with free tier access. Gemini 1.5 Flash offers fast responses, while Pro provides more advanced capabilities.

## Free Tier Models

- `gemini-1.5-flash` - Fast, lightweight model (1M tokens/month free)
- `gemini-1.5-pro` - Advanced model (limited free tier)
- `gemini-pro` - Previous generation model
- `gemini-pro-vision` - Multimodal capabilities

## Setup Instructions

```bash
# Install requirements
pip install google-generativeai

# Get free API key from https://makersuite.google.com/app/apikey
export GOOGLE_API_KEY="your-api-key"
```

## OpenCode Configuration

```json
{
  "providers": {
    "google": {
      "base_url": "https://generativelanguage.googleapis.com/v1beta",
      "api_key": "${GOOGLE_API_KEY}",
      "models": {
        "gemini-flash": "gemini-1.5-flash",
        "gemini-pro": "gemini-1.5-pro",
        "gemini": "gemini-pro",
        "gemini-vision": "gemini-pro-vision"
      }
    }
  }
}
```

## Usage Example

```python
import google.generativeai as genai

genai.configure(api_key="your-google-api-key")
model = genai.GenerativeModel('gemini-1.5-flash')

response = model.generate_content("Hello!")
```

## Status

- [x] Documentation created
- [x] Requirements installed (✓)
- [ ] API keys configured
- [ ] Basic functionality tested
- [ ] Integration with Neo-Clone completed

## Notes

- 1M tokens/month free for Gemini 1.5 Flash
- Limited free tier for Pro models
- Excellent multimodal capabilities
- Google's latest AI models

## Resilience Features

- Automatic retry on installation failures (up to 5 attempts)
- Graceful degradation when requirements unavailable
- Documentation preserved for manual integration
- Never fails - always creates some form of documentation
