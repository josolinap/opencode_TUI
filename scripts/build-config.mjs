#!/usr/bin/env node

/**
 * Build OpenCode Configuration
 * Generates JSON file from modular structure for TUI compatibility
 */

import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, "..")

// Import modular configuration
import openCodeConfig from "../config/index.js"

/**
 * Build configuration for TUI
 */
function buildConfig() {
  console.log("🔧 Building OpenCode Configuration for TUI...")

  // Create the exact structure expected by TUI
  const tuiConfig = {
    $schema: "https://opencode.ai/config.json",
    tools: openCodeConfig.tools,
    permission: openCodeConfig.permission,
    provider: openCodeConfig.providers,
    model: openCodeConfig.models?.default,
  }

  // Write to root as opencode.json (for TUI)
  const outputPath = path.join(projectRoot, "opencode.json")
  fs.writeFileSync(outputPath, JSON.stringify(tuiConfig, null, 2), "utf8")

  console.log(`✅ Configuration built: ${outputPath}`)

  // Also create a backup
  const backupPath = path.join(projectRoot, "opencode-modular-backup.json")
  fs.writeFileSync(backupPath, JSON.stringify(tuiConfig, null, 2), "utf8")
  console.log(`✅ Backup created: ${backupPath}`)

  // Generate statistics
  const stats = {
    providers: Object.keys(tuiConfig.provider || {}).length,
    totalModels: Object.values(tuiConfig.provider || {}).reduce(
      (sum, provider) => sum + Object.keys(provider.models || {}).length,
      0,
    ),
    defaultModel: tuiConfig.model,
    generatedAt: new Date().toISOString(),
  }

  console.log("\n📊 Build Statistics:")
  console.log(`   Providers: ${stats.providers}`)
  console.log(`   Total Models: ${stats.totalModels}`)
  console.log(`   Default Model: ${stats.defaultModel}`)
  console.log(`   Generated: ${stats.generatedAt}`)

  return tuiConfig
}

/**
 * Verify the built configuration
 */
function verifyConfig(config) {
  console.log("\n🔍 Verifying built configuration...")
  console.log("Config keys:", Object.keys(config))

  const requiredFields = ["$schema", "tools", "permission", "provider"]
  const missingFields = requiredFields.filter((field) => !config[field])

  if (missingFields.length > 0) {
    console.error(`❌ Missing required fields: ${missingFields.join(", ")}`)
    return false
  }

  // Verify providers
  let validProviders = 0
  for (const [name, provider] of Object.entries(config.provider || {})) {
    if (provider.api && provider.env && provider.models) {
      validProviders++
    } else {
      console.warn(`⚠️  Provider ${name} missing required fields`)
    }
  }

  console.log(`✅ ${validProviders}/${Object.keys(config.provider || {}).length} providers valid`)
  console.log("✅ Configuration verification passed")

  return true
}

// Main build process
async function main() {
  try {
    console.log("🚀 OpenCode Configuration Builder")
    console.log("===================================\n")

    const config = buildConfig()
    const isValid = verifyConfig(config)

    if (isValid) {
      console.log("\n🎉 Build successful!")
      console.log("   The configuration is ready for OpenCode TUI.")
      console.log("\n📝 Usage:")
      console.log("   - TUI will automatically use opencode.json")
      console.log("   - Modular structure remains in config/ directory")
      console.log("   - Run this script again to rebuild after changes")
    } else {
      console.error("\n❌ Build failed. Please check the errors above.")
      process.exit(1)
    }
  } catch (error) {
    console.error("❌ Build error:", error.message)
    process.exit(1)
  }
}

// Run the build
main().catch(console.error)
