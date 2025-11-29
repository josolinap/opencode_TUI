/**
 * Cohere Provider Configuration
 * Command models and embeddings
 */

export const cohereProvider = {
  api: "https://api.cohere.ai/v1",
  env: ["COHERE_API_KEY"],
  models: {
    command: "command",
    "command-light": "command-light",
    "command-nightly": "command-nightly",
    embeddings: "embed-english-v3.0",
  },
}

export default cohereProvider
