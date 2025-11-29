/**
 * OpenCode Configuration - Modular Structure
 * Main entry point that combines all configuration modules
 */

import { baseConfig } from "./base.js"
import { providers } from "./providers/index.js"
import { modelAliases } from "./models/index.js"

/**
 * Complete OpenCode configuration
 * Reconstructs the original JSON structure from modular components
 */
const openCodeConfig = {
  ...baseConfig,
  providers,
  models: modelAliases,
}

export default openCodeConfig

// Also export individual components for flexibility
export { baseConfig, providers, modelAliases }

// For backward compatibility, export the config as both default and named export
export { openCodeConfig }
