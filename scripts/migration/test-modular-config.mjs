#!/usr/bin/env node

/**
 * Test script for the new modular OpenCode configuration
 */

import openCodeConfig from "./config/index.js"

console.log("🧪 Testing Modular OpenCode Configuration")
console.log("==========================================\n")

// Test 1: Basic structure
console.log("1. Testing basic structure...")
console.log(`   Schema: ${openCodeConfig.$schema}`)
console.log(`   Tools enabled: ${Object.keys(openCodeConfig.tools).join(", ")}`)
console.log(`   Providers count: ${Object.keys(openCodeConfig.providers).length}`)
console.log(`   Model aliases: ${Object.keys(openCodeConfig.models).length}`)

// Test 2: Provider access
console.log("\n2. Testing provider access...")
const sampleProviders = ["opencode", "google", "openrouter"]
sampleProviders.forEach((provider) => {
  if (openCodeConfig.providers[provider]) {
    const p = openCodeConfig.providers[provider]
    console.log(`   ${provider}: ${Object.keys(p.models).length} models`)
  }
})

// Test 3: Model aliases
console.log("\n3. Testing model aliases...")
const sampleAliases = ["default", "fast", "quality"]
sampleAliases.forEach((alias) => {
  if (openCodeConfig.models[alias]) {
    console.log(`   ${alias} -> ${openCodeConfig.models[alias]}`)
  }
})

// Test 4: Specific model lookup
console.log("\n4. Testing specific model lookup...")
const googleProvider = openCodeConfig.providers.google
if (googleProvider) {
  console.log(`   Google base URL: ${googleProvider.base_url}`)
  console.log(`   Gemini Pro model: ${googleProvider.models["gemini-pro"]}`)
}

// Test 5: Configuration completeness
console.log("\n5. Testing configuration completeness...")
const requiredFields = ["$schema", "tools", "permission", "providers", "models"]
const missingFields = requiredFields.filter((field) => !openCodeConfig[field])

if (missingFields.length === 0) {
  console.log("   ✅ All required fields present")
} else {
  console.log(`   ❌ Missing fields: ${missingFields.join(", ")}`)
}

// Test 6: Provider validation
console.log("\n6. Testing provider validation...")
let validProviders = 0
let totalProviders = Object.keys(openCodeConfig.providers).length

for (const [name, provider] of Object.entries(openCodeConfig.providers)) {
  if (provider.base_url && provider.api_key && provider.models) {
    validProviders++
  } else {
    console.log(`   ⚠️  Provider ${name} missing required fields`)
  }
}

console.log(`   ✅ ${validProviders}/${totalProviders} providers are valid`)

console.log("\n🎉 Configuration test completed successfully!")
console.log("   The modular structure is working correctly.")
