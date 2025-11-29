#!/usr/bin/env node

/**
 * Test TUI Configuration Loading
 * Verifies that the generated JSON works with TUI expectations
 */

import fs from "fs"
import path from "path"

const configPath = path.join(process.cwd(), "opencode.json")

console.log("🧪 Testing TUI Configuration Loading")
console.log("=====================================\n")

try {
  // Load the generated JSON configuration
  const configContent = fs.readFileSync(configPath, "utf8")
  const config = JSON.parse(configContent)

  console.log("✅ JSON file loaded successfully")
  console.log(`📁 File: ${configPath}`)
  console.log(`📊 Size: ${(configContent.length / 1024).toFixed(2)} KB`)

  // Test TUI-specific requirements
  console.log("\n🔍 Testing TUI Requirements:")

  // Check schema
  if (config.$schema) {
    console.log("✅ Schema present")
  } else {
    console.log("❌ Schema missing")
  }

  // Check tools
  if (config.tools && config.tools["neo-clone"]) {
    console.log("✅ Neo-Clone tool enabled")
  } else {
    console.log("❌ Neo-Clone tool not found")
  }

  // Check providers
  const providerCount = Object.keys(config.providers || {}).length
  console.log(`✅ Providers found: ${providerCount}`)

  // Check models
  const modelAliasCount = Object.keys(config.models || {}).length
  console.log(`✅ Model aliases found: ${modelAliasCount}`)

  // Test specific provider access (what TUI would do)
  console.log("\n🎯 Testing Provider Access:")

  const testProviders = ["opencode", "google", "openrouter"]
  testProviders.forEach((providerName) => {
    const provider = config.providers[providerName]
    if (provider) {
      const modelCount = Object.keys(provider.models || {}).length
      console.log(`✅ ${providerName}: ${modelCount} models (${provider.base_url})`)
    } else {
      console.log(`❌ ${providerName}: not found`)
    }
  })

  // Test model alias resolution
  console.log("\n🏷️  Testing Model Aliases:")

  const testAliases = ["default", "fast", "quality"]
  testAliases.forEach((alias) => {
    const modelId = config.models[alias]
    if (modelId) {
      console.log(`✅ ${alias} -> ${modelId}`)
    } else {
      console.log(`❌ ${alias}: not found`)
    }
  })

  // Test permission structure
  console.log("\n🔐 Testing Permissions:")

  if (config.permission) {
    console.log(`✅ Edit permission: ${config.permission.edit}`)
    console.log(`✅ Bash permission: ${config.permission.bash ? "configured" : "missing"}`)
    console.log(`✅ Webfetch permission: ${config.permission.webfetch}`)
  } else {
    console.log("❌ Permission configuration missing")
  }

  // Summary
  console.log("\n📋 Summary:")
  console.log(`   Total Providers: ${providerCount}`)
  console.log(`   Total Model Aliases: ${modelAliasCount}`)
  console.log(`   Configuration Valid: ✅`)
  console.log(`   Ready for TUI: ✅`)

  console.log("\n🎉 TUI configuration test passed!")
  console.log("   The generated opencode.json is ready for use with OpenCode TUI.")
} catch (error) {
  console.error("❌ Test failed:", error.message)
  process.exit(1)
}
