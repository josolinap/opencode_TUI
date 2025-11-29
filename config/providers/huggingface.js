/**
 * HuggingFace Provider Configuration
 * Open source models and embeddings
 */

export const huggingfaceProvider = {
  api: "https://api-inference.huggingface.co/models",
  env: ["HF_API_KEY"],
  models: {
    "dialogpt-medium": "microsoft/DialoGPT-medium",
    distilbert: "distilbert-base-uncased",
    "bart-cnn": "facebook/bart-large-cnn",
    gpt2: "gpt2",
    minilm: "sentence-transformers/all-MiniLM-L6-v2",
  },
}

export default huggingfaceProvider
