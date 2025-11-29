#!/usr/bin/env node

/**
 * OpenCode Configuration Migration Script
 * Migrates from monolithic JSON to modular structure
 */

import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, "..")

// Configuration paths
const originalConfigPath = path.join(projectRoot, "opencode_with_free_models.json")
const configDir = path.join(projectRoot, "config")
const newConfigPath = path.join(configDir, "index.js")

/**
 * Load and parse the original configuration
 */
function loadOriginalConfig() {
  try {
    const content = fs.readFileSync(originalConfigPath, "utf8")
    return JSON.parse(content)
  } catch (error) {
    console.error("❌ Error loading original config:", error.message)
    process.exit(1)
  }
}

/**
 * Load the new modular configuration
 */
async function loadNewConfig() {
  try {
    // Dynamic import to handle ES modules
    const module = await import(`file://${newConfigPath}`)
    return module.default
  } catch (error) {
    console.error("❌ Error loading new config:", error.message)
    process.exit(1)
  }
}

/**
 * Compare two configurations
 */
function compareConfigs(original, newConfig) {
  const errors = []
  const warnings = []

  // Compare base properties
  if (original.$schema !== newConfig.$schema) {
    errors.push("Schema mismatch")
  }

  if (JSON.stringify(original.tools) !== JSON.stringify(newConfig.tools)) {
    errors.push("Tools configuration mismatch")
  }

  if (JSON.stringify(original.permission) !== JSON.stringify(newConfig.permission)) {
    errors.push("Permission configuration mismatch")
  }

  // Compare providers
  const originalProviders = original.providers || {}
  const newProviders = newConfig.providers || {}

  const originalProviderKeys = Object.keys(originalProviders).sort()
  const newProviderKeys = Object.keys(newProviders).sort()

  if (JSON.stringify(originalProviderKeys) !== JSON.stringify(newProviderKeys)) {
    errors.push(`Provider count mismatch: original=${originalProviderKeys.length}, new=${newProviderKeys.length}`)
  }

  // Check each provider
  for (const providerName of originalProviderKeys) {
    if (!newProviders[providerName]) {
      errors.push(`Missing provider: ${providerName}`)
      continue
    }

    const originalProvider = originalProviders[providerName]
    const newProvider = newProviders[providerName]

    if (originalProvider.base_url !== newProvider.base_url) {
      errors.push(`Provider ${providerName}: base_url mismatch`)
    }

    if (originalProvider.api_key !== newProvider.api_key) {
      errors.push(`Provider ${providerName}: api_key mismatch`)
    }

    const originalModels = originalProvider.models || {}
    const newModels = newProvider.models || {}

    const originalModelKeys = Object.keys(originalModels).sort()
    const newModelKeys = Object.keys(newModels).sort()

    if (JSON.stringify(originalModelKeys) !== JSON.stringify(newModelKeys)) {
      errors.push(`Provider ${providerName}: model count mismatch`)
    }

    for (const modelKey of originalModelKeys) {
      if (originalModels[modelKey] !== newModels[modelKey]) {
        errors.push(`Provider ${providerName}: model ${modelKey} value mismatch`)
      }
    }
  }

  // Compare model aliases
  const originalModels = original.models || {}
  const newModels = newConfig.models || {}

  const originalModelKeys = Object.keys(originalModels).sort()
  const newModelKeys = Object.keys(newModels).sort()

  if (JSON.stringify(originalModelKeys) !== JSON.stringify(newModelKeys)) {
    warnings.push(`Model aliases count mismatch: original=${originalModelKeys.length}, new=${newModelKeys.length}`)
  }

  for (const modelKey of originalModelKeys) {
    if (originalModels[modelKey] !== newModels[modelKey]) {
      warnings.push(`Model alias ${modelKey} value mismatch`)
    }
  }

  return { errors, warnings }
}

/**
 * Generate statistics
 */
function generateStats(config) {
  const providers = config.providers || {}
  const totalProviders = Object.keys(providers).length
  const totalModels = Object.values(providers).reduce((sum, provider) => {
    return sum + Object.keys(provider.models || {}).length
  }, 0)
  const modelAliases = Object.keys(config.models || {}).length

  return {
    totalProviders,
    totalModels,
    modelAliases,
    averageModelsPerProvider: (totalModels / totalProviders).toFixed(1),
  }
}

/**
 * Main migration function
 */
async function main() {
  console.log("🔧 OpenCode Configuration Migration")
  console.log("=====================================\n")

  // Load configurations
  console.log("📂 Loading configurations...")
  const originalConfig = loadOriginalConfig()
  const newConfig = await loadNewConfig()
  console.log("✅ Configurations loaded successfully\n")

  // Compare configurations
  console.log("🔍 Comparing configurations...")
  const { errors, warnings } = compareConfigs(originalConfig, newConfig)

  if (errors.length > 0) {
    console.log("\n❌ Errors found:")
    errors.forEach((error) => console.log(`   - ${error}`))
  }

  if (warnings.length > 0) {
    console.log("\n⚠️  Warnings found:")
    warnings.forEach((warning) => console.log(`   - ${warning}`))
  }

  if (errors.length === 0 && warnings.length === 0) {
    console.log("✅ Perfect match! No differences found.")
  }

  // Generate statistics
  console.log("\n📊 Configuration Statistics:")
  const originalStats = generateStats(originalConfig)
  const newStats = generateStats(newConfig)

  console.log("\nOriginal Configuration:")
  console.log(`   Providers: ${originalStats.totalProviders}`)
  console.log(`   Models: ${originalStats.totalModels}`)
  console.log(`   Model Aliases: ${originalStats.modelAliases}`)
  console.log(`   Avg Models/Provider: ${originalStats.averageModelsPerProvider}`)

  console.log("\nNew Modular Configuration:")
  console.log(`   Providers: ${newStats.totalProviders}`)
  console.log(`   Models: ${newStats.totalModels}`)
  console.log(`   Model Aliases: ${newStats.modelAliases}`)
  console.log(`   Avg Models/Provider: ${newStats.averageModelsPerProvider}`)

  // File structure analysis
  console.log("\n📁 Modular Structure Benefits:")
  console.log("   ✅ Separated concerns (providers vs models)")
  console.log("   ✅ Individual provider files for easy maintenance")
  console.log("   ✅ Model categorization and aliases")
  console.log("   ✅ Reusable components")
  console.log("   ✅ Better organization and scalability")

  // Success message
  if (errors.length === 0) {
    console.log("\n🎉 Migration successful!")
    console.log("   The modular structure maintains full compatibility")
    console.log("   while providing better organization and maintainability.")
    console.log("\n📝 Next steps:")
    console.log("   1. Test the new configuration with your application")
    console.log("   2. Update any imports to use the new modular structure")
    console.log("   3. Consider removing the original monolithic file")
  } else {
    console.log("\n❌ Migration failed. Please fix the errors above.")
    process.exit(1)
  }
}

// Run the migration
main().catch(console.error)
