# LLM Integration: Cohere (Free Tier)

## Provider Details

- **Provider**: Cohere
- **API Type**: api_key
- **Models**: command, command-light, command-nightly
- **Requirements**: cohere
- **Integration Date**: 2025-11-29T10:52:30.000000

## Description

Cohere provides powerful language models with free tier access. Excellent for text generation, classification, and embeddings.

## Free Tier Models

- `command` - Main instruction-following model
- `command-light` - Faster, lighter version
- `command-nightly` - Latest experimental features
- `embed-english-v3.0` - Text embeddings

## Setup Instructions

```bash
# Install requirements
pip install cohere

# Get free API key from https://dashboard.cohere.com/api-keys
export COHERE_API_KEY="your-api-key"
```

## OpenCode Configuration

```json
{
  "providers": {
    "cohere": {
      "base_url": "https://api.cohere.ai/v1",
      "api_key": "${COHERE_API_KEY}",
      "models": {
        "command": "command",
        "command-light": "command-light",
        "command-nightly": "command-nightly",
        "embeddings": "embed-english-v3.0"
      }
    }
  }
}
```

## Usage Example

```python
import cohere

client = cohere.Client(api_key="your-cohere-api-key")

response = client.generate(
    model="command",
    prompt="Write a hello world program in Python",
    max_tokens=100
)
```

## Status

- [x] Documentation created
- [ ] Requirements installed (✓)
- [ ] API keys configured
- [ ] Basic functionality tested
- [ ] Integration with Neo-Clone completed

## Notes

- 100 free calls/month
- Excellent for instruction following
- Strong text generation capabilities
- Also provides embeddings and classification

## Resilience Features

- Automatic retry on installation failures (up to 5 attempts)
- Graceful degradation when requirements unavailable
- Documentation preserved for manual integration
- Never fails - always creates some form of documentation
