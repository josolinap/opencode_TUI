/**
 * Perplexity Provider Configuration
 * Search-enabled models
 */

export const perplexityProvider = {
  api: "https://api.perplexity.ai",
  env: ["PERPLEXITY_API_KEY"],
  models: {
    "sonar-small": "llama-3.1-sonar-small-128k-online",
    "sonar-large": "llama-3.1-sonar-large-128k-online",
    "sonar-medium": "llama-3.1-sonar-medium-128k-online",
  },
}

export default perplexityProvider
