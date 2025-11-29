/**
 * OpenAI Provider Configuration
 * Official OpenAI models
 */

export const openaiProvider = {
  api: "https://api.openai.com/v1",
  env: ["OPENAI_API_KEY"],
  models: {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
  },
}

export default openaiProvider
