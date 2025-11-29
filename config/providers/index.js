/**
 * Provider Registry
 * Aggregates all provider configurations
 */

import opencodeProvider from "./opencode.js"
import openrouterProvider from "./openrouter.js"
import groqProvider from "./groq.js"
import googleProvider from "./google.js"
import huggingfaceProvider from "./huggingface.js"
import cohereProvider from "./cohere.js"
import replicateProvider from "./replicate.js"
import perplexityProvider from "./perplexity.js"
import openaiProvider from "./openai.js"
import anthropicProvider from "./anthropic.js"
import togetherProvider from "./together.js"

export const providers = {
  opencode: opencodeProvider,
  openrouter: openrouterProvider,
  groq: groqProvider,
  google: googleProvider,
  huggingface: huggingfaceProvider,
  cohere: cohereProvider,
  replicate: replicateProvider,
  perplexity: perplexityProvider,
  openai: openaiProvider,
  anthropic: anthropicProvider,
  together: togetherProvider,
}

export default providers
