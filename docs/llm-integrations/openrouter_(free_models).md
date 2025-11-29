# LLM Integration: OpenRouter (Free Models)

## Provider Details

- **Provider**: OpenRouter
- **API Type**: api_key
- **Models**: 500+ models with free variants
- **Requirements**: openai
- **Integration Date**: 2025-11-29T10:51:30.000000

## Description

OpenRouter provides unified access to 500+ AI models from various providers with free tier options using the `:free` variant suffix.

## Free Model Examples

Add `:free` suffix to any model ID for free access:

### Popular Free Models

- `meta-llama/llama-3.2-3b-instruct:free`
- `meta-llama/llama-3.1-8b-instruct:free`
- `mistralai/mistral-7b-instruct:free`
- `microsoft/wizardlm-2-8x22b:free`
- `google/gemma-2-9b-it:free`
- `qwen/qwen-2.5-7b-instruct:free`
- `anthropic/claude-3-haiku:free`

## Setup Instructions

```bash
# Install requirements
pip install openai

# Get free API key from https://openrouter.ai/keys
export OPENROUTER_API_KEY="your-api-key"
```

## OpenCode Configuration

```json
{
  "providers": {
    "openrouter": {
      "base_url": "https://openrouter.ai/api/v1",
      "api_key": "${OPENROUTER_API_KEY}",
      "models": {
        "llama-3.2-3b-free": "meta-llama/llama-3.2-3b-instruct:free",
        "llama-3.1-8b-free": "meta-llama/llama-3.1-8b-instruct:free",
        "mistral-7b-free": "mistralai/mistral-7b-instruct:free",
        "gemma-2-9b-free": "google/gemma-2-9b-it:free",
        "claude-haiku-free": "anthropic/claude-3-haiku:free"
      }
    }
  }
}
```

## Usage Example

```python
import openai

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-api-key"
)

response = client.chat.completions.create(
    model="meta-llama/llama-3.2-3b-instruct:free",
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

- Free models may have rate limits
- Some models may have availability restrictions
- Check https://openrouter.ai/models for current free model list
- Use `:free` suffix for free variants

## Resilience Features

- Automatic retry on installation failures (up to 5 attempts)
- Graceful degradation when requirements unavailable
- Documentation preserved for manual integration
- Never fails - always creates some form of documentation
