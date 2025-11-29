/**
 * Replicate Provider Configuration
 * Replicate hosted models
 */

export const replicateProvider = {
  api: "https://api.replicate.com/v1",
  env: ["REPLICATE_API_TOKEN"],
  models: {
    "llama-2-70b": "meta/llama-2-70b-chat",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
    "llama-3-70b": "meta/meta-llama-3-70b-instruct",
    arctic: "snowflake/snowflake-arctic-instruct",
    "stable-diffusion": "stability-ai/stable-diffusion",
  },
}

export default replicateProvider
