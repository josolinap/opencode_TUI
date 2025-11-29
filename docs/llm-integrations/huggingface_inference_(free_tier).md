# LLM Integration: Hugging Face Inference (Free Tier)

## Provider Details

- **Provider**: Hugging Face Inference
- **API Type**: api_key
- **Models**: 100,000+ models available
- **Requirements**: requests, huggingface_hub
- **Integration Date**: 2025-11-29T10:52:15.000000

## Description

Hugging Face provides free inference access to thousands of models through their Inference API. Perfect for experimenting with open-source models.

## Free Tier Models

- `microsoft/DialoGPT-medium` - Conversational AI
- `distilbert-base-uncased` - Text classification
- `facebook/bart-large-cnn` - Summarization
- `gpt2` - Text generation
- `microsoft/DialoGPT-large` - Advanced conversation
- `sentence-transformers/all-MiniLM-L6-v2` - Embeddings

## Setup Instructions

```bash
# Install requirements
pip install huggingface_hub requests

# Get free API key from https://huggingface.co/settings/tokens
export HF_API_KEY="your-api-key"
```

## OpenCode Configuration

```json
{
  "providers": {
    "huggingface": {
      "base_url": "https://api-inference.huggingface.co/models",
      "api_key": "${HF_API_KEY}",
      "models": {
        "dialogpt-medium": "microsoft/DialoGPT-medium",
        "distilbert": "distilbert-base-uncased",
        "bart-cnn": "facebook/bart-large-cnn",
        "gpt2": "gpt2",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2"
      }
    }
  }
}
```

## Usage Example

```python
from huggingface_hub import InferenceClient

client = InferenceClient(api_key="your-hf-token")

response = client.text_generation(
    model="microsoft/DialoGPT-medium",
    prompt="Hello, how are you?"
)
```

## Status

- [x] Documentation created
- [ ] Requirements installed (✓)
- [ ] API keys configured
- [ ] Basic functionality tested
- [ ] Integration with Neo-Clone completed

## Notes

- 30,000 requests/month free tier
- Access to 100,000+ models
- Perfect for open-source model experimentation
- Some models may have longer response times

## Resilience Features

- Automatic retry on installation failures (up to 5 attempts)
- Graceful degradation when requirements unavailable
- Documentation preserved for manual integration
- Never fails - always creates some form of documentation
