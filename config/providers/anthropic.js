/**
 * Anthropic Provider Configuration
 * Claude models
 */

export const anthropicProvider = {
  api: "https://api.anthropic.com/v1",
  env: ["ANTHROPIC_API_KEY"],
  models: {
    "claude-3-haiku": "claude-3-haiku-20240307",
  },
}

export default anthropicProvider
