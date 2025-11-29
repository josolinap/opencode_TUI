# LLM Integration: Replicate (Free Tier)

## Provider Details

- **Provider**: Replicate
- **API Type**: api_key
- **Models**: 1000+ open-source models
- **Requirements**: replicate
- **Integration Date**: 2025-11-29T10:52:45.000000

## Description

Replicate provides access to thousands of open-source models with free tier credits. Perfect for running models like Llama, Stable Diffusion, and more.

## Free Tier Models

- `meta/llama-2-70b-chat` - Llama 2 70B chat model
- `stability-ai/stable-diffusion` - Image generation
- `mistralai/mixtral-8x7b-instruct-v0.1` - Mixtral MoE model
- `meta/meta-llama-3-70b-instruct` - Llama 3 70B
- `snowflake/snowflake-arctic-instruct` - Arctic model

## Setup Instructions

```bash
# Install requirements
pip install replicate

# Get free API key from https://replicate.com/account
export REPLICATE_API_TOKEN="your-api-token"
```

## OpenCode Configuration

```json
{
  "providers": {
    "replicate": {
      "base_url": "https://api.replicate.com/v1",
      "api_key": "${REPLICATE_API_TOKEN}",
      "models": {
        "llama-2-70b": "meta/llama-2-70b-chat",
        "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
        "llama-3-70b": "meta/meta-llama-3-70b-instruct",
        "arctic": "snowflake/snowflake-arctic-instruct",
        "stable-diffusion": "stability-ai/stable-diffusion"
      }
    }
  }
}
```

## Usage Example

```python
import replicate

client = replicate.Client(api_token="your-replicate-token")

response = client.run(
    "meta/llama-2-70b-chat",
    input={"prompt": "Hello, how are you?"}
)
```

## Status

- [x] Documentation created
- [ ] Requirements installed (✓)
- [ ] API keys configured
- [ ] Basic functionality tested
- [ ] Integration with Neo-Clone completed

## Notes

- $5 free credits for new users
- Access to 1000+ open-source models
- Pay-per-use after free credits
- Excellent for experimenting with latest models

## Resilience Features

- Automatic retry on installation failures (up to 5 attempts)
- Graceful degradation when requirements unavailable
- Documentation preserved for manual integration
- Never fails - always creates some form of documentation
