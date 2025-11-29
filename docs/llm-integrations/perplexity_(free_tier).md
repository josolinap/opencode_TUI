# LLM Integration: Perplexity (Free Tier)

## Provider Details

- **Provider**: Perplexity AI
- **API Type**: api_key
- **Models**: llama-3.1-sonar-small-128k-online, llama-3.1-sonar-large-128k-online
- **Requirements**: openai
- **Integration Date**: 2025-11-29T10:53:00.000000

## Description

Perplexity provides models with real-time web search capabilities. Perfect for up-to-date information and research tasks.

## Free Tier Models

- `llama-3.1-sonar-small-128k-online` - Fast model with web search
- `llama-3.1-sonar-large-128k-online` - More capable model with web search
- `llama-3.1-sonar-medium-128k-online` - Balanced model with web search

## Setup Instructions

```bash
# Install requirements
pip install openai

# Get free API key from https://www.perplexity.ai/settings/api
export PERPLEXITY_API_KEY="your-api-key"
```

## OpenCode Configuration

```json
{
  "providers": {
    "perplexity": {
      "base_url": "https://api.perplexity.ai",
      "api_key": "${PERPLEXITY_API_KEY}",
      "models": {
        "sonar-small": "llama-3.1-sonar-small-128k-online",
        "sonar-large": "llama-3.1-sonar-large-128k-online",
        "sonar-medium": "llama-3.1-sonar-medium-128k-online"
      }
    }
  }
}
```

## Usage Example

```python
import openai

client = openai.OpenAI(
    api_key="your-perplexity-api-key",
    base_url="https://api.perplexity.ai"
)

response = client.chat.completions.create(
    model="llama-3.1-sonar-small-128k-online",
    messages=[
        {"role": "system", "content": "Be precise and concise."},
        {"role": "user", "content": "What are the latest developments in AI?"}
    ]
)
```

## Status

- [x] Documentation created
- [ ] Requirements installed (✓)
- [ ] API keys configured
- [ ] Basic functionality tested
- [ ] Integration with Neo-Clone completed

## Notes

- Real-time web search capabilities
- 128K context window
- OpenAI-compatible API
- Excellent for research and current information

## Resilience Features

- Automatic retry on installation failures (up to 5 attempts)
- Graceful degradation when requirements unavailable
- Documentation preserved for manual integration
- Never fails - always creates some form of documentation
