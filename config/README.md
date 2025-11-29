# OpenCode Configuration - Modular Structure

This directory contains the modular OpenCode configuration that replaces the monolithic `opencode_with_free_models.json` file.

## 🎯 Problem Solved

The original configuration was a "Model Monster" - a single 131-line JSON file containing:

- 11 different providers
- 39 different models
- 9 model aliases
- All mixed together in one file

This made it difficult to:

- Maintain individual provider configurations
- Add new providers or models
- Understand the structure at a glance
- Test individual components

## 🏗️ New Modular Structure

```
config/
├── README.md                 # This documentation
├── index.js                  # Main entry point
├── base.js                   # Base configuration (tools, permissions)
├── providers/                 # Provider configurations
│   ├── index.js              # Provider registry
│   ├── opencode.js           # OpenCode free models
│   ├── openrouter.js         # OpenRouter free tier
│   ├── groq.js               # Groq fast inference
│   ├── google.js              # Google Gemini models
│   ├── huggingface.js        # HuggingFace models
│   ├── cohere.js             # Cohere models
│   ├── replicate.js          # Replicate models
│   ├── perplexity.js         # Perplexity search models
│   ├── openai.js             # OpenAI models
│   ├── anthropic.js          # Anthropic Claude models
│   └── together.js           # Together AI models
└── models/                   # Model configurations
    ├── index.js              # Model aliases
    └── categories.js         # Model categories
```

## 📊 Configuration Statistics

- **Providers**: 11
- **Total Models**: 39
- **Model Aliases**: 9
- **Average Models per Provider**: 3.5

## 🚀 Usage

### Basic Usage

```javascript
import openCodeConfig from "./config/index.js"

// Access the complete configuration
console.log(openCodeConfig.providers.google.models["gemini-pro"])
// Output: "gemini-1.5-pro"

// Access model aliases
console.log(openCodeConfig.models.default)
// Output: "big-pickle"
```

### Individual Components

```javascript
// Import only providers
import { providers } from "./config/index.js"

// Import only model aliases
import { modelAliases } from "./config/index.js"

// Import base configuration
import { baseConfig } from "./config/index.js"
```

### Provider-Specific Access

```javascript
import { providers } from "./config/index.js"

// Access Google provider
const google = providers.google
console.log(google.base_url) // https://generativelanguage.googleapis.com/v1beta
console.log(google.models) // { gemini-flash: "...", gemini-pro: "...", ... }
```

## 🔧 Adding New Providers

1. Create a new file in `config/providers/`:

```javascript
// config/providers/newprovider.js
export const newproviderProvider = {
  base_url: "https://api.newprovider.com/v1",
  api_key: "${NEWPROVIDER_API_KEY}",
  models: {
    "model-name": "internal-model-id",
  },
}

export default newproviderProvider
```

2. Add to `config/providers/index.js`:

```javascript
import newproviderProvider from "./newprovider.js"

export const providers = {
  // ... existing providers
  newprovider: newproviderProvider,
}
```

3. The provider is automatically available in the main configuration!

## 🏷️ Model Categories

The `config/models/categories.js` file organizes models by capability:

- **fast**: Quick inference models
- **quality**: High-quality models for complex tasks
- **free**: Free tier models
- **multimodal**: Models with vision capabilities
- **search**: Models with web search
- **conversation**: Dialogue-optimized models
- **coding**: Code generation models
- **embeddings**: Text embedding models

## ✅ Benefits

### Before (Monolithic)

- ❌ Single 131-line file
- ❌ Hard to maintain
- ❌ Difficult to add providers
- ❌ Mixed concerns
- ❌ Poor readability

### After (Modular)

- ✅ Separated concerns
- ✅ Easy provider management
- ✅ Better organization
- ✅ Reusable components
- ✅ Improved maintainability
- ✅ Better testing capabilities
- ✅ Clear documentation

## 🧪 Testing

Run the migration verification script:

```bash
node scripts/migrate-opencode-config.mjs
```

Run the configuration test:

```bash
node test-modular-config.mjs
```

## 🔄 Migration Status

- ✅ All providers migrated
- ✅ All models preserved
- ✅ All aliases maintained
- ✅ Full compatibility verified
- ✅ Tests passing

## 📝 Next Steps

1. Update your application to use the new modular configuration
2. Test with your existing workflows
3. Consider removing the original `opencode_with_free_models.json` file
4. Enjoy the improved maintainability!

## 🤝 Contributing

When adding new providers or models:

1. Follow the existing file naming convention
2. Include proper JSDoc comments
3. Update this README if needed
4. Run the test scripts to verify compatibility

---

**Result**: The "Model Monster" has been tamed! 🎉
From a single 131-line monolith to a clean, modular, maintainable structure.
