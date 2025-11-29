/**
 * Model Categories
 * Categorizes models by their capabilities and use cases
 */

export const modelCategories = {
  // Speed-focused models
  fast: {
    description: "Fast inference models for quick responses",
    models: ["llama-3.1-8b", "gemma-7b", "gemma2-9b", "claude-3-haiku"],
  },

  // Quality-focused models
  quality: {
    description: "High-quality models for complex tasks",
    models: ["gemini-pro", "llama-3.1-70b", "mixtral-8x7b", "command"],
  },

  // Free tier models
  free: {
    description: "Free tier models with no cost",
    models: ["big-pickle", "grok-code", "gpt-5-nano", "llama-3.2-3b-free", "mistral-7b-free"],
  },

  // Multimodal models
  multimodal: {
    description: "Models that support images and vision",
    models: ["gemini-vision", "gemini-flash", "gemini-pro"],
  },

  // Search-enabled models
  search: {
    description: "Models with real-time web search capabilities",
    models: ["sonar-small", "sonar-medium", "sonar-large"],
  },

  // Conversation models
  conversation: {
    description: "Models optimized for dialogue and chat",
    models: ["dialogpt-medium", "claude-3-haiku", "gpt-3.5-turbo"],
  },

  // Coding models
  coding: {
    description: "Models optimized for code generation and analysis",
    models: ["llama-3.1-8b", "mixtral-8x7b", "command-light"],
  },

  // Embedding models
  embeddings: {
    description: "Models for text embeddings and semantic search",
    models: ["minilm", "embeddings"],
  },
}

export default modelCategories
