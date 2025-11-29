/**
 * Base OpenCode Configuration
 * Contains shared configuration that applies across all providers
 */

export const baseConfig = {
  $schema: "https://opencode.ai/config.json",
  tools: {
    "neo-clone": true,
  },
  permission: {
    edit: "allow",
    bash: {
      "*": "allow",
    },
    webfetch: "allow",
  },
}

export default baseConfig
