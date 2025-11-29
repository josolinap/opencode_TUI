/**
 * Together Provider Configuration
 * Together AI hosted models
 */

export const togetherProvider = {
  api: "https://api.together.xyz/v1",
  env: ["TOGETHER_API_KEY"],
  models: {
    "llama-2-70b": "togethercomputer/llama-2-70b-chat",
  },
}

export default togetherProvider
