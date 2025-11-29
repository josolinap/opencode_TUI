/**
 * Google Provider Configuration
 * Gemini models
 */

export const googleProvider = {
  api: "https://generativelanguage.googleapis.com/v1beta",
  env: ["GOOGLE_API_KEY"],
  models: {
    "gemini-flash": "gemini-1.5-flash",
    "gemini-pro": "gemini-1.5-pro",
    gemini: "gemini-pro",
    "gemini-vision": "gemini-pro-vision",
  },
}

export default googleProvider
