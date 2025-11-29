/**
 * OpenCode Provider Configuration
 * Free OpenCode models
 */

export const opencodeProvider = {
  api: "http://localhost:3000",
  env: [],
  models: {
    "big-pickle": {
      id: "big-pickle",
      name: "Big Pickle",
      release_date: "2024-01-01",
      attachment: false,
      reasoning: false,
      temperature: true,
      tool_call: true,
      cost: { input: 0, output: 0 },
      limit: { context: 128000, output: 4096 },
    },
    "grok-code": {
      id: "grok-code",
      name: "Grok Code",
      release_date: "2024-01-01",
      attachment: false,
      reasoning: false,
      temperature: true,
      tool_call: true,
      cost: { input: 0, output: 0 },
      limit: { context: 128000, output: 4096 },
    },
    "gpt-5-nano": {
      id: "gpt-5-nano",
      name: "GPT-5 Nano",
      release_date: "2024-01-01",
      attachment: false,
      reasoning: false,
      temperature: true,
      tool_call: true,
      cost: { input: 0, output: 0 },
      limit: { context: 128000, output: 4096 },
    },
  },
}

export default opencodeProvider
