/**
 * OpenRouter Provider Configuration
 * Free tier models from OpenRouter
 */

export const openrouterProvider = {
  api: "https://openrouter.ai/api/v1",
  env: ["OPENROUTER_API_KEY"],
  models: {
    "llama-3.2-3b-free": "meta-llama/llama-3.2-3b-instruct:free",
    "llama-3.1-8b-free": "meta-llama/llama-3.1-8b-instruct:free",
    "mistral-7b-free": "mistralai/mistral-7b-instruct:free",
    "gemma-2-9b-free": "google/gemma-2-9b-it:free",
    "claude-haiku-free": "anthropic/claude-3-haiku:free",
    "qwen-7b-free": "qwen/qwen-2.5-7b-instruct:free",
    "wizardlm-8b-free": "microsoft/wizardlm-2-8x22b:free",
  },
}

export default openrouterProvider
