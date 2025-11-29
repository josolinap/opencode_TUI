/**
 * Groq Provider Configuration
 * Fast inference models
 */

export const groqProvider = {
  api: "https://api.groq.com/openai/v1",
  env: ["GROQ_API_KEY"],
  models: {
    "llama-3.1-8b": "llama-3.1-8b-instant",
    "llama-3.1-70b": "llama-3.1-70b-versatile",
    "mixtral-8x7b": "mixtral-8x7b-32768",
    "gemma-7b": "gemma-7b-it",
    "gemma2-9b": "gemma2-9b-it",
  },
}

export default groqProvider
