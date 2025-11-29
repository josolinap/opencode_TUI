#!/usr/bin/env node

/**
 * Verify All Models Configuration
 * Comprehensive test of all models in the configuration
 */

import fs from "fs"
import path from "path"

const configPath = path.join(process.cwd(), "opencode.json")

console.log("🔍 Verifying All Models Configuration")
console.log("====================================\n")

try {
  // Load configuration
  const config = JSON.parse(fs.readFileSync(configPath, "utf8"))

  let totalModels = 0
  let validProviders = 0
  const issues = []

  console.log("📊 Provider Analysis:")
  console.log("====================")

  // Analyze each provider
  for (const [providerName, provider] of Object.entries(config.providers)) {
    const models = provider.models || {}
    const modelCount = Object.keys(models).length
    totalModels += modelCount

    console.log(`\n🔌 ${providerName.toUpperCase()}`)
    console.log(`   URL: ${provider.base_url}`)
    console.log(`   API Key: ${provider.api_key.includes("${") ? "Environment Variable" : "Direct"}`)
    console.log(`   Models: ${modelCount}`)

    // Validate provider structure
    let providerValid = true

    if (!provider.base_url) {
      issues.push(`${providerName}: Missing base_url`)
      providerValid = false
    }

    if (!provider.api_key) {
      issues.push(`${providerName}: Missing api_key`)
      providerValid = false
    }

    if (!models || Object.keys(models).length === 0) {
      issues.push(`${providerName}: No models defined`)
      providerValid = false
    }

    if (providerValid) {
      validProviders++
      console.log(`   ✅ Valid provider`)

      // List models
      for (const [modelName, modelId] of Object.entries(models)) {
        console.log(`      - ${modelName} → ${modelId}`)
      }
    } else {
      console.log(`   ❌ Invalid provider`)
    }
  }

  console.log("\n🏷️  Model Aliases Analysis:")
  console.log("===========================")

  const aliases = config.models || {}
  let validAliases = 0

  for (const [aliasName, modelId] of Object.entries(aliases)) {
    // Check if the aliased model exists in any provider
    let modelExists = false
    for (const provider of Object.values(config.providers)) {
      if (provider.models && provider.models[modelId]) {
        modelExists = true
        break
      }
    }

    if (modelExists) {
      console.log(`   ✅ ${aliasName} → ${modelId}`)
      validAliases++
    } else {
      console.log(`   ❌ ${aliasName} → ${modelId} (model not found)`)
      issues.push(`Alias ${aliasName}: Model ${modelId} not found in any provider`)
    }
  }

  console.log("\n📋 Summary Statistics:")
  console.log("======================")
  console.log(`   Total Providers: ${Object.keys(config.providers).length}`)
  console.log(`   Valid Providers: ${validProviders}`)
  console.log(`   Total Models: ${totalModels}`)
  console.log(`   Total Aliases: ${Object.keys(aliases).length}`)
  console.log(`   Valid Aliases: ${validAliases}`)

  console.log("\n🔐 Security Analysis:")
  console.log("====================")

  let secureProviders = 0
  for (const [providerName, provider] of Object.entries(config.providers)) {
    const usesEnvVar = provider.api_key.includes("${")
    if (usesEnvVar) {
      console.log(`   ✅ ${providerName}: Uses environment variables`)
      secureProviders++
    } else {
      console.log(`   ⚠️  ${providerName}: Uses direct API key`)
    }
  }

  console.log(`   Secure Providers: ${secureProviders}/${Object.keys(config.providers).length}`)

  console.log("\n🎯 Specialized Models:")
  console.log("====================")

  // Categorize models by capability
  const categories = {
    "Free Models": [],
    "Fast Models": [],
    "Quality Models": [],
    "Search Models": [],
    "Multimodal Models": [],
  }

  for (const [providerName, provider] of Object.entries(config.providers)) {
    for (const [modelName, modelId] of Object.entries(provider.models)) {
      if (providerName === "opencode" || modelName.includes("-free")) {
        categories["Free Models"].push(`${providerName}/${modelName}`)
      }
      if (providerName === "groq" || modelName.includes("fast") || modelName.includes("instant")) {
        categories["Fast Models"].push(`${providerName}/${modelName}`)
      }
      if (providerName === "google" || modelName.includes("pro") || modelName.includes("70b")) {
        categories["Quality Models"].push(`${providerName}/${modelName}`)
      }
      if (providerName === "perplexity" || modelName.includes("sonar")) {
        categories["Search Models"].push(`${providerName}/${modelName}`)
      }
      if (modelName.includes("vision") || modelName.includes("multimodal")) {
        categories["Multimodal Models"].push(`${providerName}/${modelName}`)
      }
    }
  }

  for (const [category, models] of Object.entries(categories)) {
    if (models.length > 0) {
      console.log(`\n   ${category}: ${models.length}`)
      models.slice(0, 3).forEach((model) => console.log(`      - ${model}`))
      if (models.length > 3) {
        console.log(`      ... and ${models.length - 3} more`)
      }
    }
  }

  // Final verdict
  console.log("\n🎯 Final Verdict:")
  console.log("=================")

  if (issues.length === 0) {
    console.log("✅ All configurations are valid!")
    console.log("✅ All models are properly configured!")
    console.log("✅ All aliases resolve to existing models!")
    console.log("✅ Configuration is ready for production use!")
  } else {
    console.log("❌ Issues found:")
    issues.forEach((issue) => console.log(`   - ${issue}`))
  }

  console.log("\n🚀 Ready for TUI: " + (issues.length === 0 ? "✅ YES" : "❌ NO"))
} catch (error) {
  console.error("❌ Verification failed:", error.message)
  process.exit(1)
}
